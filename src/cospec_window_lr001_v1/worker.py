#!/usr/bin/env python3
"""One isolated GPU task. Epoch checkpoints and resumable full-TE inference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import fcntl
import gc
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from torch.nn import functional as F
from common import (read_json, write_json, recipe_id, job_dir, identity, memory_rss_mib,
                    profile_key, SafeStop, batch_count)
from models import configure_imports, build_model, augmentation
from data import Patches
configure_imports()
from external_common import seed_all, capture_rng, restore_rng, state_hash
from external_data import ObservationSampler


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class Progress:
    def __init__(self, path, rid):
        self.path, self.rid = Path(path), rid
        self.last = 0.
        self.values = dict(recipe_id=rid, **identity())

    def update(self, force=False, **values):
        self.values.update(values)
        now = time.monotonic()
        if force or now - self.last >= 2:
            memory = dict(cuda_reserved_mib=torch.cuda.memory_reserved() / 2**20,
                          cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20) \
                     if torch.cuda.is_available() else dict(cuda_reserved_mib=0., cuda_peak_reserved_mib=0.)
            self.values.update(updated_unix=time.time(), rss_mib=memory_rss_mib(), **memory)
            write_json(self.path, self.values)
            self.last = now


def merged_batches(count, batch, generator):
    order = torch.randperm(count, generator=generator).tolist()
    parts = [order[i:i + batch] for i in range(0, count, batch)]
    if len(parts) > 1 and len(parts[-1]) == 1:
        parts[-2].extend(parts.pop())
    return parts


def train(model, config, job, directory, progress, stop, stop_after_epoch=None):
    meta = config['data'][job['dataset']]
    settings = config['train']
    device = config['device']
    dataset = Patches(meta, 'tr', job['patch'], job['seed'] + 11)
    generator = torch.Generator().manual_seed(job['seed'] + 22)
    sampler = ObservationSampler(meta['shape'][-1], job['seed'] + 44, [c['mask'] for c in meta['cases']])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    checkpoint = directory / 'last.pt'
    history, start = {}, 0
    rid = recipe_id(config, job)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['recipe_id'] != rid:
            raise ValueError('检查点协议不符；拒绝复用旧分阶段学习率权重')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'])
        generator.set_state(saved['loader_rng'].cpu())
        dataset.rng.bit_generator.state = saved['rotation_rng']
        sampler.rng.bit_generator.state = saved['mask_rng']
        history, start = saved['history'], saved['epoch']
        del saved
    if any(v['lr'] != .001 for v in history.values()) or any(p['lr'] != .001 for p in optimizer.param_groups):
        raise ValueError('历史实际学习率不是全程0.001')
    write_json(directory / 'history.json', history)
    steps = batch_count(len(dataset), settings['batch'])
    progress.update(force=True, phase='train', epoch=start, steps_per_epoch=steps,
                    train_steps_done=start * steps, train_steps_total=settings['epochs'] * steps)
    for epoch in range(start + 1, settings['epochs'] + 1):
        if stop():
            raise SafeStop()
        model.train()
        began = time.perf_counter()
        full_total = missing_total = 0.
        seen = 0
        for index, indices in enumerate(merged_batches(len(dataset), settings['batch'], generator), 1):
            x, y = dataset.batch(indices)
            x, y = x.to(device), y.to(device)
            full = torch.ones((len(y), meta['shape'][-1]), dtype=torch.bool, device=device)
            drawn = sampler.draw(len(y), device)
            missing = drawn if augmentation(job) else full
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for mask in (full, missing):
                logits = model(x, mask)['logits']
                ce = F.cross_entropy(logits, y)
                if not torch.isfinite(ce):
                    raise FloatingPointError('训练损失包含NaN/Inf')
                (ce / 2).backward()
                losses.append(float(ce.detach()))
                del logits, ce
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 15., error_if_nonfinite=True)
            optimizer.step()
            full_total += losses[0] * len(y)
            missing_total += losses[1] * len(y)
            seen += len(y)
            progress.update(phase='train', epoch=epoch, batch=index, batches=steps,
                            train_steps_done=(epoch - 1) * steps + index, last_loss=sum(losses) / 2)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        history[str(epoch)] = dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'],
                                  loss=(full_total + missing_total) / (2 * seen), full_ce=full_total / seen,
                                  missing_ce=missing_total / seen, last_gradient_norm=float(norm),
                                  seconds=time.perf_counter() - began, training_centers_seen=seen)
        atomic_torch(checkpoint, dict(recipe_id=rid, epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), rng=capture_rng(), loader_rng=generator.get_state(),
                     rotation_rng=copy.deepcopy(dataset.rng.bit_generator.state),
                     mask_rng=copy.deepcopy(sampler.rng.bit_generator.state), history=history))
        write_json(directory / 'history.json', history)
        progress.update(force=True, epoch=epoch, epoch_saved=epoch,
                        recent_epoch_seconds=history[str(epoch)]['seconds'])
        print(f'epoch {epoch}/{settings["epochs"]} lr=0.001 loss={history[str(epoch)]["loss"]:.6f} checkpoint_saved', flush=True)
        if stop() or stop_after_epoch == epoch:
            raise SafeStop()
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return history


class Metrics:
    """Streaming versions of the archived metrics; no full probability cube."""
    def __init__(self, classes, state=None):
        self.k = classes
        self.s = state or dict(matrix=np.zeros((classes, classes), np.int64).tolist(), count=0,
                               nll=0., brier=0., high_error=0, bins_n=[0]*15,
                               bins_conf=[0.]*15, bins_correct=[0]*15)

    def add(self, truth, prob):
        prob = np.asarray(prob, np.float64)
        truth = np.asarray(truth, np.int64)
        if prob.shape != (len(truth), self.k) or not np.isfinite(prob).all():
            raise ValueError('测试概率形状/数值异常')
        if np.any(prob < 0) or not np.allclose(prob.sum(1), 1, atol=1e-4):
            raise ValueError('无效预测概率')
        pred, confidence = prob.argmax(1), prob.max(1)
        matrix = np.asarray(self.s['matrix'], np.int64)
        np.add.at(matrix, (truth, pred), 1)
        self.s['matrix'] = matrix.tolist()
        target = np.eye(self.k)[truth]
        self.s['count'] += len(truth)
        self.s['nll'] += float(-np.log(np.maximum(prob[np.arange(len(truth)), truth], 1e-12)).sum())
        self.s['brier'] += float(((prob - target)**2).sum())
        correct = pred == truth
        self.s['high_error'] += int(((confidence >= .9) & ~correct).sum())
        for i, (lo, hi) in enumerate(zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:])):
            rows = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
            self.s['bins_n'][i] += int(rows.sum())
            self.s['bins_conf'][i] += float(confidence[rows].sum())
            self.s['bins_correct'][i] += int(correct[rows].sum())
        return pred, confidence

    def finish(self):
        matrix = np.asarray(self.s['matrix'], np.int64)
        n = self.s['count']
        support, predicted = matrix.sum(1), matrix.sum(0)
        recall = np.divide(matrix.diagonal(), support, out=np.zeros(self.k), where=support > 0)
        precision = np.divide(matrix.diagonal(), predicted, out=np.zeros(self.k), where=predicted > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(self.k), where=precision + recall > 0)
        oa = float(matrix.trace() / n)
        expected = float(np.dot(support.astype(float), predicted) / n**2)
        ece = sum(abs(c - t) / n for c, t in zip(self.s['bins_conf'], self.s['bins_correct']))
        return dict(count=n, oa_percent=100*oa, aa_percent=100*float(recall[support > 0].mean()),
                    kappa=(oa-expected)/(1-expected) if expected < 1 else None,
                    macro_f1_present_classes=float(f1[support > 0].mean()), nll=self.s['nll']/n,
                    brier=self.s['brier']/n, ece_15_bins=ece, high_confidence_error_fraction=self.s['high_error']/n,
                    per_class_accuracy_percent=[float(100*r) if count else None for r, count in zip(recall, support)],
                    class_support=support.tolist(), confusion_matrix=matrix.tolist())


def evaluate(model, config, job, directory, model_sha, progress, stop):
    meta = config['data'][job['dataset']]
    dataset = Patches(meta, 'te', job['patch'])
    batch = config['train']['eval_batch']
    steps = (len(dataset) + batch - 1) // batch
    rid = recipe_id(config, job)
    metrics = []
    model.eval()
    with torch.inference_mode():
        for case_index, case in enumerate(meta['cases']):
            prediction = directory / (case['name'] + '.npz')
            metric_file = directory / (case['name'] + '.metrics.json')
            if prediction.exists() and metric_file.exists():
                old = read_json(metric_file)
                if (old['recipe_id'], old['model_sha256'], old['mask_sha256'], old['count']) != (rid, model_sha, case['sha256'], len(dataset)):
                    raise ValueError('已有测试结果协议不符')
                metrics.append(old)
                continue
            partial = directory / 'partial' / case['name']
            partial.mkdir(parents=True, exist_ok=True)
            marker = partial / 'cursor.json'
            identity_fields = dict(recipe_id=rid, model_sha256=model_sha, mask_sha256=case['sha256'])
            cursor = read_json(marker) if marker.exists() else None
            if cursor and any(cursor[k] != v for k, v in identity_fields.items()):
                raise ValueError('测试断点权重或掩码不匹配')
            start = cursor['next_pixel'] if cursor else 0
            accumulator = Metrics(meta['classes'], cursor['metrics'] if cursor else None)
            if accumulator.s['count'] != start:
                raise ValueError('测试断点统计数量与像元游标不同')
            mode = 'r+' if cursor else 'w+'
            pred = np.lib.format.open_memmap(partial / 'pred.npy', mode=mode, dtype=np.int16, shape=(len(dataset),))
            conf = np.lib.format.open_memmap(partial / 'confidence.npy', mode=mode, dtype=np.float32, shape=(len(dataset),))
            if pred.shape != (len(dataset),) or conf.shape != pred.shape or not 0 <= start <= len(dataset):
                raise ValueError('测试断点尺寸异常')
            observed = torch.tensor(case['mask'], dtype=torch.bool, device=config['device'])
            began = time.perf_counter()
            for offset in range(start, len(dataset), batch):
                end = min(offset + batch, len(dataset))
                x, y = dataset.batch(range(offset, end))
                logits = model(x.to(config['device']), observed[None].expand(len(y), -1))['logits']
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('测试logits包含NaN/Inf')
                probs = logits.softmax(-1).cpu().numpy()
                predictions, confidence = accumulator.add(y.numpy(), probs)
                pred[offset:end], conf[offset:end] = predictions, confidence
                step = (end + batch - 1) // batch
                progress.update(phase='test', case=case['name'], case_index=case_index, case_count=16,
                                test_steps_done=case_index * steps + step, test_steps_total=16 * steps,
                                current_case_pixel=end, current_case_total=len(dataset))
                want_stop = stop()
                if step % 50 == 0 or end == len(dataset) or want_stop:
                    pred.flush(); conf.flush()
                    write_json(marker, dict(**identity_fields, next_pixel=end, metrics=accumulator.s))
                if want_stop:
                    raise SafeStop()
                del logits, probs, x, y
            atomic_npz(prediction, labels=dataset.labels, predictions=np.asarray(pred), confidence=np.asarray(conf),
                       positions=dataset.coords, observed=np.asarray(case['mask'], bool))
            row = dict(**identity_fields, case=case['name'], fraction=case['fraction'], repeat=case['repeat'],
                       seconds_this_segment=time.perf_counter()-began, **accumulator.finish())
            if row['count'] != meta['test_count'] or row['class_support'] != meta['class_support']:
                raise ValueError('测试像元未完整覆盖')
            write_json(metric_file, row)
            metrics.append(row)
            del pred, conf
            import shutil
            shutil.rmtree(partial)
            progress.update(force=True, phase='test', test_steps_done=(case_index + 1)*steps)
            if config['device'].startswith('cuda'):
                torch.cuda.empty_cache()
    return metrics


def memory_probe(config, job, path, stop=lambda: False):
    """Warm up and measure train32 and eval128, including low-keep operator paths."""
    seed_all(job['seed'])
    meta = config['data'][job['dataset']]
    dataset = Patches(meta, 'tr', job['patch'], job['seed'] + 11)
    model = build_model(config, job).to(config['device'])
    train_batch = min(len(dataset), config['train']['batch'])
    x, y = dataset.batch(range(train_batch))
    x, y = x.to(config['device']), y.to(config['device'])
    sampler = ObservationSampler(meta['shape'][-1], job['seed']+44, [c['mask'] for c in meta['cases']])
    missing = sampler.draw(train_batch, config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    if config['device'].startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    times = []
    model.train()
    for iteration in range(3):
        if stop():
            raise SafeStop()
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for mask in (torch.ones_like(missing), missing):
            if stop():
                raise SafeStop()
            loss = F.cross_entropy(model(x, mask)['logits'], y)
            if not torch.isfinite(loss):
                raise FloatingPointError('预检训练损失不有限')
            (loss/2).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 15., error_if_nonfinite=True)
        optimizer.step()
        if config['device'].startswith('cuda'):
            torch.cuda.synchronize()
        if iteration:
            times.append(time.perf_counter()-began)
    train_peak = torch.cuda.max_memory_reserved()/2**20 if config['device'].startswith('cuda') else 0.
    optimizer.zero_grad(set_to_none=True)
    del optimizer, x, y, loss
    gc.collect()
    if config['device'].startswith('cuda'):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    testset = Patches(meta, 'te', job['patch'])
    count = min(len(testset), config['train']['eval_batch'])
    x, y = testset.batch(range(count))
    x = x.to(config['device'])
    model.eval()
    eval_times = []
    with torch.inference_mode():
        for name in ('full', 'keep10_r1'):
            mask = torch.tensor(next(c['mask'] for c in meta['cases'] if c['name'] == name), dtype=torch.bool,
                                device=config['device'])[None].expand(count, -1)
            for repetition in range(3):
                if stop():
                    raise SafeStop()
                began = time.perf_counter()
                out = model(x, mask)['logits']
                if out.shape != (count, meta['classes']) or not torch.isfinite(out).all():
                    raise ValueError('显存预检的测试输出异常')
                if config['device'].startswith('cuda'):
                    torch.cuda.synchronize()
                if repetition:
                    eval_times.append(time.perf_counter()-began)
    eval_peak = torch.cuda.max_memory_reserved()/2**20 if config['device'].startswith('cuda') else 0.
    import resource
    result = dict(complete=True, profile_key=profile_key(config, job), job=job,
                  train_peak_reserved_mib=train_peak, eval_peak_reserved_mib=eval_peak,
                  train_batch_seconds=float(np.median(times)), eval_batch_seconds=float(np.median(eval_times)),
                  rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  torch=str(torch.__version__), **identity())
    if stop():
        raise SafeStop()
    write_json(path, result)
    return result


def run(config, job, stop, factory=build_model):
    directory = job_dir(config, job)
    directory.mkdir(parents=True, exist_ok=True)
    rid = recipe_id(config, job)
    progress = Progress(directory / 'progress.json', rid)
    with (directory / 'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (directory / 'outcome.json').unlink(missing_ok=True)
        result_file = directory / 'result.json'
        if result_file.exists():
            result = read_json(result_file)
            if result['recipe_id'] != rid:
                raise ValueError('已有完成结果来自不同协议')
            if result.get('complete'):
                return
        seed_all(job['seed'])
        progress.update(force=True, phase='initializing')
        model = factory(config, job).to(config['device'])
        initial = dict(state_sha256=state_hash(model.state_dict()),
                       parameters=sum(p.numel() for p in model.parameters()), recipe_id=rid)
        if hasattr(model, 'backbone_state'):
            initial['common_backbone_sha256'] = state_hash({k: v for k, v in model.backbone_state().items()
                                                          if not k.startswith(('scan_orders', 'scan_inverse'))})
        previous = directory / 'initialization.json'
        if previous.exists() and read_json(previous) != initial:
            raise ValueError('初始化改变，不混用旧断点')
        write_json(previous, initial)
        history = train(model, config, job, directory, progress, stop)
        sha = state_hash(model.state_dict())
        model.eval()
        metrics = evaluate(model, config, job, directory, sha, progress, stop)
        from exports import export_all
        rng = capture_rng()
        try:
            export_all(model, config, job, directory, sha, progress, stop)
        finally:
            restore_rng(rng)
        write_json(result_file, dict(complete=True, recipe_id=rid, job=job, scope=config['scope'],
                   model_sha256=sha, completed_epochs=config['train']['epochs'], metrics=metrics,
                   train_seconds=sum(r['seconds'] for r in history.values()),
                   initial_state_sha256=initial['state_sha256'], runtime=dict(torch=str(torch.__version__)),
                   source_sha256=config['source_sha256'], lr_policy='constant_0.001'))
        progress.update(force=True, phase='completed')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--probe', type=Path)
    args = p.parse_args()
    config = read_json(args.config)
    job = config['jobs'][args.index]
    torch.set_num_threads(1)
    stopped = [False]
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    stop = lambda: stopped[0] or (Path(config['output'])/'STOP').exists()
    directory = job_dir(config, job)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        if args.probe:
            with args.probe.with_suffix('.lock').open('a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                memory_probe(config, job, args.probe, stop)
        else:
            run(config, job, stop)
        return 0
    except SafeStop:
        write_json(directory/'outcome.json', dict(state='stopped', recipe_id=recipe_id(config, job)))
        return 130
    except Exception as error:
        oom = isinstance(error, torch.OutOfMemoryError) or 'out of memory' in str(error).lower()
        write_json(directory/'outcome.json', dict(state='oom' if oom else 'failed', error=str(error),
                   recipe_id=recipe_id(config, job), traceback=traceback.format_exc()))
        traceback.print_exc()
        return 75 if oom else 1


if __name__ == '__main__':
    raise SystemExit(main())
