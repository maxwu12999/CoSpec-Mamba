#!/usr/bin/env python3
"""Resume only the two failed MiM jobs, with an audited zero-sigma boundary.

Upload this file beside cospec_full_lr001_v1, then run: python FILE start.
The core source/config/LR/masks stay intact. The numerical extension is recorded
separately in checkpoints, results, the recovery audit, and the result archive.
"""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile

BASE = Path(__file__).resolve().parent
REVISION = 'mim_zero_sigma_uniform_v1'
EXPECTED_SOURCE = '40a0f0ccaa744c3d0c748caacda4a53f0718d6c3a4a0429a26d70ff9e1c2f841'
EXPECTED_PROTOCOL = '3812432c7c2e0e1263921712ca2606756c1fe826a2e0a2c7d59271fd815bf9b4'
LONGKOU_SHA = '4e6daa9c09bbe3e9d540b1b6ece6295ff7e533e7b45793139631c7e9549fa264'
TARGETS = {('LongKou', 'MiM_v2', 2, 15), ('Salinas', 'MiM_v2', 3, 15)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def note(message):
    print(time.strftime('[%m-%d %H:%M:%S] ') + str(message), flush=True)


def key(job):
    return tuple(job[n] for n in ('dataset', 'model', 'seed', 'patch'))


def core_imports(core):
    core = Path(core).resolve()
    if not (core / 'worker.py').is_file():
        raise FileNotFoundError('找不到完整实验源码：' + str(core))
    sys.path.insert(0, str(core))
    import common
    if common.ROOT != core:
        raise RuntimeError('导入了其他目录的 common，拒绝运行')
    import results
    return common, results


def gaussian_safe(torch, sequence, mode='center'):
    """Keep positive sigma unchanged; define uniform weights when sigma == 0.

    The safe denominator is selected BEFORE division. At equal vectors, norm's
    zero subgradient and this constant branch give finite, zero gradients.
    NaN/Inf inputs are not sanitized: the ordinary worker finite checks fail.
    """
    length = sequence.shape[1]
    reference = sequence[:, (length + 1) // 2, :] if mode == 'center' else sequence[:, -1, :]
    distances = torch.norm(sequence - reference.unsqueeze(1), dim=2)
    sigma = distances.mean(dim=1, keepdim=True)
    zero = sigma == 0
    denominator = torch.where(zero, torch.ones_like(sigma), sigma)
    weights = torch.exp(-0.5 * (distances / denominator) ** 2)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return weights, zero


def upstream_gaussian(core, torch):
    tree = ast.parse((Path(core) / 'frozen/external/vendor/mim/MiM_v2.py').read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'T_Mamba')
    func = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                              and n.name == 'gaussian_mask_vector'))
    func.decorator_list = []
    space = dict(torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[])),
                 '<archived MiM gaussian>', 'exec'), space)
    return space['gaussian_mask_vector']


def numerical_check(core, device='cpu'):
    import torch
    original = upstream_gaussian(core, torch)
    for mode in ('center', 'last'):
        # Deterministic inputs: this check does not consume experiment RNG.
        raw = torch.arange(4 * 25 * 8, device=device, dtype=torch.float32).reshape(4, 25, 8)
        raw = torch.sin(raw * .371)
        a, b = raw.clone().requires_grad_(), raw.clone().requires_grad_()
        left, right = original(a, mode), gaussian_safe(torch, b, mode)[0]
        if not torch.equal(left, right):
            raise AssertionError('正常输入的高斯权重发生变化')
        upstream = torch.cos(torch.arange(left.numel(), device=device).reshape_as(left).float())
        (left * upstream).sum().backward()
        (right * upstream).sum().backward()
        if not torch.equal(a.grad, b.grad):
            raise AssertionError('正常输入的高斯梯度发生变化')
        mixed = raw.clone()
        mixed[1] = 1
        mixed.requires_grad_()
        fixed, zero = gaussian_safe(torch, mixed, mode)
        if int(zero.sum()) != 1 or not torch.isfinite(fixed).all():
            raise AssertionError('零距离边界检查失败')
        torch.testing.assert_close(fixed[1], torch.full_like(fixed[1], 1 / 25), rtol=0, atol=0)
        (fixed * upstream).sum().backward()
        if not torch.isfinite(mixed.grad).all() or bool(mixed.grad[1].count_nonzero()):
            raise AssertionError('零距离梯度检查失败')
        if not torch.equal(fixed[[0, 2, 3]], left[[0, 2, 3]]):
            raise AssertionError('混合批次的正常样本发生变化')
    return dict(device=device, positive_sigma_forward_exact=True,
                positive_sigma_gradient_exact=True, mixed_batch_exact=True,
                zero_sigma_uniform=True, zero_sigma_gradient_finite=True)


class GaussianGuard:
    """Patch dynamic MiM classes in one worker, without modifying vendor files."""
    def __init__(self, torch, model, origin, previous=None):
        self.torch, self.model = torch, model
        self.origin = origin
        self.base = copy.deepcopy((previous or {}).get('counters', {}))
        self.counts, self.handles, self.originals = {}, [], {}
        self.location = 'T_Mamba'

    def __enter__(self):
        classes = set()
        for name, module in self.model.named_modules():
            if type(module).__name__ != 'T_Mamba':
                continue
            classes.add(type(module))
            def enter(_module, _args, name=name):
                self.location = name or 'T_Mamba'
            self.handles.append(module.register_forward_pre_hook(enter))
        if not classes:
            raise RuntimeError('模型内未找到 T_Mamba，未应用任何数值修复')
        for cls in classes:
            self.originals[cls] = cls.gaussian_mask_vector
            def guarded(sequence, type='center'):
                weights, zero = gaussian_safe(self.torch, sequence, type)
                where = self.location + '/' + ('train' if self.model.training else 'test')
                if where not in self.counts:
                    self.counts[where] = self.torch.zeros(2, dtype=self.torch.int64, device=sequence.device)
                # Keep counters on-device; no per-forward .item()/GPU barrier.
                self.counts[where][0].add_(1)
                self.counts[where][1].add_(zero.detach().sum())
                return weights
            cls.gaussian_mask_vector = staticmethod(guarded)
        return self

    def metadata(self):
        counters = copy.deepcopy(self.base)
        for name, values in self.counts.items():
            calls, rows = values.detach().cpu().tolist()
            old = counters.get(name, dict(calls=0, zero_sigma_rows=0))
            counters[name] = dict(calls=old['calls'] + calls, zero_sigma_rows=old['zero_sigma_rows'] + rows)
        return dict(revision=REVISION, script_sha256=sha(__file__),
                    kind='numerical_boundary_extension', positive_sigma='original formula unchanged',
                    zero_sigma='uniform weights; zero autograd gradient at equal vectors',
                    origin=self.origin, counters=counters,
                    zero_sigma_rows=sum(v['zero_sigma_rows'] for v in counters.values()))

    def __exit__(self, *_):
        for cls, original in self.originals.items():
            cls.gaussian_mask_vector = staticmethod(original)
        for handle in self.handles:
            handle.remove()


def finite_tree(torch, value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(torch, v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(torch, v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def inspect_checkpoint(config, job, common, enforce_origin=True):
    import torch
    path = common.job_dir(config, job) / 'last.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('recipe_id') != common.recipe_id(config, job):
        raise ValueError('检查点协议不同：' + str(path))
    epoch = saved.get('epoch')
    if not isinstance(epoch, int) or not 0 <= epoch <= config['train']['epochs']:
        raise ValueError('检查点轮次异常')
    if not all(finite_tree(torch, saved[n]) for n in ('model', 'optimizer', 'history')):
        raise ValueError('检查点含有 NaN/Inf；未开始恢复')
    if set(saved['history']) != {str(n) for n in range(1, epoch + 1)}:
        raise ValueError('检查点训练历史轮次不完整')
    if (any(h['lr'] != .001 for h in saved['history'].values()) or
            any(g['lr'] != .001 for g in saved['optimizer']['param_groups'])):
        raise ValueError('检查点不是全程固定0.001')
    previous = saved.get('numerical_repair')
    checkpoint_sha = sha(path)
    if previous:
        if previous.get('revision') != REVISION or previous.get('script_sha256') != sha(__file__):
            raise ValueError('检查点来自其他数值修复版本')
        origin = previous['origin']
    else:
        if enforce_origin and job['dataset'] == 'LongKou' and (epoch != 83 or checkpoint_sha != LONGKOU_SHA):
            raise ValueError('LongKou 原始83轮断点与诊断记录不一致')
        origin = dict(job=job, checkpoint_epoch=epoch, checkpoint_sha256=checkpoint_sha,
                      protocol_id=config['protocol_id'])
    return dict(epoch=epoch, checkpoint_sha256=checkpoint_sha, origin=origin, previous=previous)


def protected_files(config, completed, common):
    """JSON content hashes + sizes/mtimes for all files of completed runs."""
    root = Path(config['output'])
    snapshot = {}
    for index in completed:
        for path in sorted(common.job_dir(config, config['jobs'][index]).rglob('*')):
            if not path.is_file() or path.suffix in ('.tmp', '.lock'):
                continue
            stat = path.stat()
            value = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if path.suffix in ('.json', '.csv', '.txt'):
                value['sha256'] = sha(path)
            snapshot[str(path.relative_to(root))] = value
    return snapshot


def verify_protected(root, snapshot):
    for relative, old in snapshot.items():
        path = Path(root) / relative
        st = path.stat()
        if st.st_size != old['bytes'] or st.st_mtime_ns != old['mtime_ns']:
            raise RuntimeError('已完成任务文件发生变化：' + relative)
        if 'sha256' in old and sha(path) != old['sha256']:
            raise RuntimeError('已完成任务结果发生变化：' + relative)


def validate_config(root, common, results, strict=True):
    root = Path(root).resolve()
    config = read(root / 'config.json')
    if Path(config['output']).resolve() != root:
        raise ValueError('config.json 的输出路径不匹配')
    if strict:
        if common.source_hash() != EXPECTED_SOURCE or config['source_sha256'] != EXPECTED_SOURCE:
            raise ValueError('源码版本与已定位的实验不同；原文件未改动')
        if config['protocol_id'] != EXPECTED_PROTOCOL or len(config['jobs']) != 210:
            raise ValueError('不是本次210项固定学习率实验')
        if config['train']['epochs'] != 100 or config['train']['batch'] != 32 or config['train']['eval_batch'] != 128:
            raise ValueError('训练或测试设置与已定位实验不同')
        if config['train']['lr'] != .001 or config['train']['lr_policy'] != 'constant':
            raise ValueError('配置不是全程固定0.001')
        common.verify_frozen()
    done, pending = [], []
    for i, job in enumerate(config['jobs']):
        if results.complete(config, job):
            artifact = common.job_dir(config, job) / 'artifacts.json'
            if not artifact.is_file() or not read(artifact).get('complete'):
                raise ValueError('已完成任务导出记录不完整：' + common.label(job))
            done.append(i)
        else:
            if key(job) not in TARGETS or job.get('group') != 'external105':
                raise ValueError('发现两项目标之外的未完成任务，拒绝启动：' + common.label(job))
            pending.append(i)
    if len({key(j) for j in config['jobs']}) != len(config['jobs']):
        raise ValueError('任务清单有重复')
    plans = {str(i): inspect_checkpoint(config, config['jobs'][i], common, enforce_origin=strict) for i in pending}
    return config, done, pending, plans


def child_run(root, core, index, session, factory=None, enforce_origin=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    common, results = core_imports(core)
    import worker
    from models import build_model
    torch.set_num_threads(1)
    config = read(Path(root) / 'config.json')
    job = config['jobs'][index]
    if key(job) not in TARGETS:
        raise ValueError('修复子进程只能加载两项目标')
    info = inspect_checkpoint(config, job, common, enforce_origin=enforce_origin)
    directory = common.job_dir(config, job)
    session = Path(session)
    stopped = [False]
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    stop = lambda: stopped[0] or (Path(root) / 'STOP').exists() or (Path(root) / 'mim_recovery.cancel').exists()
    guard_box = []
    original_save, original_write = worker.atomic_torch, worker.write_json
    report_file = session / ('job_' + str(index) + '.json')
    report = dict(job=job, start_epoch=info['epoch'], origin=info['origin'], state='running')

    def patched_factory(cfg, j):
        model = (factory or build_model)(cfg, j)
        guard = GaussianGuard(torch, model, info['origin'], info['previous'])
        guard.__enter__()
        guard_box.append(guard)
        return model

    def save_checkpoint(path, value):
        if Path(path) == directory / 'last.pt':
            value = dict(value, numerical_repair=guard_box[0].metadata())
            rows = value['numerical_repair']['zero_sigma_rows']
            if rows:
                note(common.label(job) + f'：累计零距离样本调用数={rows}')
        original_save(path, value)

    def save_json(path, value):
        if Path(path) == directory / 'result.json':
            value = copy.deepcopy(value)
            value.setdefault('runtime', {})['numerical_repair'] = guard_box[0].metadata()
            value['runtime']['recovery_input_pipeline'] = 'original core worker'
        original_write(path, value)

    try:
        report['numerical_check'] = numerical_check(core, config['device'])
        write(report_file, report)
        worker.atomic_torch, worker.write_json = save_checkpoint, save_json
        note(common.label(job) + f'：从已保存{info["epoch"]}轮继续至100轮，学习率0.001')
        worker.run(config, job, stop, factory=patched_factory)
        if not results.complete(config, job):
            raise RuntimeError('恢复后16条件结果校验未通过')
        report['state'] = 'completed'
        return_code = 0
    except common.SafeStop:
        report['state'] = 'stopped'
        return_code = 130
    except Exception as error:
        report.update(state='oom' if isinstance(error, torch.OutOfMemoryError) else 'failed',
                      error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
        return_code = 75 if report['state'] == 'oom' else 1
    finally:
        worker.atomic_torch, worker.write_json = original_save, original_write
        if guard_box:
            report['numerical_repair'] = guard_box[0].metadata()
            guard_box[0].__exit__()
        report['updated_unix'] = time.time()
        write(report_file, report)
    write(directory / 'outcome.json', dict(state=report['state'], error=report.get('error'),
          recipe_id=common.recipe_id(config, job), recovery_report=str(report_file)))
    note(common.label(job) + '：' + report['state'])
    return return_code


def backup_session(root, config, done, pending, plans, common):
    session = Path(root) / 'runtime_numerical_repair' / time.strftime('%Y%m%d_%H%M%S')
    session.mkdir(parents=True, exist_ok=False)
    for name in ('config.json', 'status.json', 'failures.json', 'summary.json'):
        if (Path(root) / name).is_file():
            dest = session / 'backup' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(root) / name, dest)
    for i in pending:
        directory = common.job_dir(config, config['jobs'][i])
        dest = session / 'backup' / common.task_key(config['jobs'][i])
        dest.mkdir(parents=True)
        for name in ('last.pt', 'history.json', 'progress.json', 'outcome.json', 'initialization.json'):
            if (directory / name).is_file():
                shutil.copy2(directory / name, dest / name)
        if sha(dest / 'last.pt') != plans[str(i)]['checkpoint_sha256']:
            raise ValueError('检查点备份校验失败')
    # JSON is deliberately selected by the existing result packager.
    write(session / 'recovery_source.json', dict(filename=Path(__file__).name,
          sha256=sha(__file__), revision=REVISION, source=Path(__file__).read_text(encoding='utf-8')))
    snapshot = protected_files(config, done, common)
    write(session / 'protected_completed_files.json', snapshot)
    write(session / 'plan.json', dict(protocol_id=config['protocol_id'], revision=REVISION,
          completed_before=len(done), jobs={str(i): dict(job=config['jobs'][i], **plans[str(i)]) for i in pending}))
    return session, snapshot


def gpu_budget(gpu):
    apps = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    if apps.stdout.strip():
        raise RuntimeError('GPU仍有计算进程，先等待它们结束；未启动补跑')
    memory = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    return float(memory.stdout.strip().splitlines()[0])


def finalize(config, snapshot, session, common, results):
    root = Path(config['output'])
    verify_protected(root, snapshot)
    summary = results.aggregate(config)
    if not summary['complete']:
        return False
    affected = []
    for job in config['jobs']:
        if key(job) in TARGETS:
            value = read(common.job_dir(config, job) / 'result.json')
            repair = value.get('runtime', {}).get('numerical_repair')
            if repair is None or repair.get('revision') != REVISION:
                raise RuntimeError('目标任务缺少数值修复记录：' + common.label(job))
            affected.append(dict(job=job, numerical_repair=repair))
    write(session / 'audit.json', dict(verified=True, protocol_id=config['protocol_id'],
          completed=summary['completed'], total=summary['total'],
          previously_completed_files_unchanged=True, affected_jobs=affected,
          interpretation='Zero-sigma extension only; positive-sigma formula is unchanged. '
                         'This is a numerical repair, not a new training seed or a speed-only change.'))
    for tag in ('maps', 'features', 'results'):
        note('重新打包：' + tag)
        path = results.package(config, tag)
        with zipfile.ZipFile(path) as archive:
            if not archive.namelist() or archive.testzip() is not None:
                raise RuntimeError('ZIP完整性校验失败：' + str(path))
    verify_protected(root, snapshot)
    return True


def controller(root, core, workers=2, gpu=0):
    common, results = core_imports(core)
    root = Path(root).resolve()
    report = dict(state='preflight', revision=REVISION, updated_unix=time.time())
    session, active, pending, logs = None, {}, [], []
    stopping = [False]
    signal.signal(signal.SIGINT, lambda *_: stopping.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))
    latest = root / 'mim_recovery.latest.json'
    with (root / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if any(common.locked(p) for p in root.glob('runs/**/run.lock')):
                raise RuntimeError('仍有正式工作进程，未开始恢复')
            config, done, pending, plans = validate_config(root, common, results)
            config_sha, source_sha = sha(root / 'config.json'), common.source_hash()
            free_mib = gpu_budget(gpu) if pending else 0
            cap = min(workers, 2 if free_mib >= 14 * 1024 else 1)
            if pending and free_mib < 6 * 1024:
                raise RuntimeError('可用显存不足6GiB，未开始恢复')
            session, snapshot = backup_session(root, config, done, pending, plans, common)
            report.update(state='running', session=str(session), completed_before=len(done),
                          max_parallel=cap, planned_jobs=[config['jobs'][i] for i in pending])
            write(latest, report)
            failures, retries = {}, {}
            old_failures = read(root / 'failures.json') if (root / 'failures.json').exists() else []
            completed = set(done)

            def status(state='running', phase='补跑MiM与16条件测试'):
                entries = []
                for i, process in active.items():
                    progress_path = common.job_dir(config, config['jobs'][i]) / 'progress.json'
                    v = read(progress_path) if progress_path.exists() else {}
                    entries.append(dict(index=i, label=common.label(config['jobs'][i]),
                                        phase=v.get('phase', 'initializing'), pid=process.pid, adopted=False))
                write(root / 'status.json', dict(state=state, phase=phase, total=len(config['jobs']),
                      completed=len(completed), active=len(active), max_parallel=cap, active_jobs=entries,
                      eta_seconds=0 if state == 'completed' else None, failed=len(failures),
                      max_observed_parallel=cap, adopted_workers=0, updated_unix=time.time(), **common.identity()))

            # Keep original failure details until the corresponding task is resolved.
            write(root / 'failure_history' / (session.name + '_before_mim_repair.json'), old_failures)
            status()
            last_tick = 0.
            stop = False
            while pending or active:
                stop = stopping[0] or (root / 'STOP').exists() or (root / 'mim_recovery.cancel').exists()
                if stop:
                    (root / 'mim_recovery.cancel').touch(exist_ok=True)
                for i, process in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    del active[i]
                    job_report_path = session / ('job_' + str(i) + '.json')
                    jr = read(job_report_path) if job_report_path.exists() else {}
                    if code == 0 and results.complete(config, config['jobs'][i]):
                        completed.add(i)
                    elif code == 75 and retries.get(i, 0) < 1 and not stop:
                        retries[i] = retries.get(i, 0) + 1
                        cap = 1
                        pending.append(i)
                        note('显存不足：等待其他补跑结束，再单路重试同一检查点')
                    elif code == 130:
                        stopping[0] = True
                        stop = True
                        (root / 'mim_recovery.cancel').touch(exist_ok=True)
                    else:
                        failures[i] = dict(job=config['jobs'][i], exit_code=code,
                                           error=jr.get('error', '补跑进程异常退出'),
                                           log=str(session / ('task_' + str(i) + '.log')))
                    unresolved = [f for f in old_failures if key(f['job']) not in
                                  {key(config['jobs'][k]) for k in completed} and
                                  key(f['job']) not in {key(fv['job']) for fv in failures.values()}]
                    write(root / 'failures.json', unresolved + list(failures.values()))
                if stop and not active:
                    break
                while pending and not stop and len(active) < cap:
                    i = pending.pop(0)
                    log = (session / ('task_' + str(i) + '.log')).open('ab', buffering=0)
                    logs.append(log)
                    env = os.environ.copy()
                    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                               CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
                    active[i] = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '_worker',
                          '--output', str(root), '--core', str(Path(core).resolve()), '--index', str(i),
                          '--session', str(session)], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                          env=env, start_new_session=True)
                    note('补跑启动：' + common.label(config['jobs'][i]))
                if time.monotonic() - last_tick >= 5:
                    status(state='stopping' if stop else 'running')
                    last_tick = time.monotonic()
                time.sleep(.5)
            if common.source_hash() != source_sha or sha(root / 'config.json') != config_sha:
                raise RuntimeError('运行期间原源码或配置发生变化，拒绝更新完成状态')
            verify_protected(root, snapshot)
            if stop:
                status(state='stopped', phase='补跑已安全停止')
                report['state'] = 'stopped'
                return_code = 130
            else:
                # Do not advertise completed until summary and all three ZIPs pass.
                unresolved = [f for f in old_failures if not results.complete(config, f['job'])
                              and key(f['job']) not in {key(v['job']) for v in failures.values()}]
                write(root / 'failures.json', unresolved + list(failures.values()))
                status(phase='汇总与打包')
                success = finalize(config, snapshot, session, common, results)
                status(state='completed' if success else 'partial', phase='全部完成' if success else 'partial')
                report['state'] = 'completed' if success else 'partial'
                return_code = 0 if success else 1
                if success:
                    write(root / 'failures.json', [])
                    note(f'{len(completed)}/{len(config["jobs"])}通过，三份ZIP校验通过；原已完成文件未改变。')
            report.update(completed=len(completed), total=len(config['jobs']))
        except Exception as error:
            traceback.print_exc()
            # If this controller fails, ask its own workers to finish a safe boundary.
            if active:
                (root / 'mim_recovery.cancel').touch(exist_ok=True)
                while any(p.poll() is None for p in active.values()):
                    time.sleep(.5)
            report.update(state='failed', error=str(error), traceback=traceback.format_exc())
            if session is not None:
                previous = read(root / 'status.json')
                previous.update(state='failed', phase='补跑或打包失败', error=str(error), active=0,
                                active_jobs=[], eta_seconds=None, updated_unix=time.time())
                write(root / 'status.json', previous)
            return_code = 1
        finally:
            for log in logs:
                log.close()
            report['updated_unix'] = time.time()
            write(latest, report)
            if session is not None:
                write(session / 'final.json', report)
    return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status', 'watch', 'stop', 'check', '_run', '_worker'))
    parser.add_argument('--core', type=Path, default=BASE / 'cospec_full_lr001_v1')
    parser.add_argument('--output', type=Path, default=BASE / 'diagnostics/COSPEC_FULL_LR001_V1')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--index', type=int)
    parser.add_argument('--session', type=Path)
    args = parser.parse_args()
    root, core = args.output.expanduser().resolve(), args.core.expanduser().resolve()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.action == '_worker':
        return child_run(root, core, args.index, args.session)
    if args.action == '_run':
        return controller(root, core, args.workers, args.gpu)
    common, results = core_imports(core)
    if args.action in ('status', 'watch'):
        from dashboard import watch
        watch(root, once=args.action == 'status')
        if args.action == 'status' and (root / 'mim_recovery.latest.json').exists():
            r = read(root / 'mim_recovery.latest.json')
            print('补跑：' + r['state'] + ('；' + r['error'] if r.get('error') else ''))
        print('补跑日志：' + str(root / 'mim_recovery.console.log'))
        return 0
    if args.action == 'stop':
        (root / 'mim_recovery.cancel').touch(exist_ok=True)
        print('已请求两项补跑保存完整轮次/测试批次后停止。')
        return 0
    if args.action == 'check':
        print(json.dumps(numerical_check(core), ensure_ascii=False, indent=2))
        return 0
    if not root.is_dir():
        raise FileNotFoundError('找不到原实验输出：' + str(root))
    with (root / 'launch.lock').open('a+') as launch:
        fcntl.flock(launch, fcntl.LOCK_EX)
        if common.locked(root / 'queue.lock'):
            print('原队列或补跑仍占有队列锁，未重复启动。')
            return 0
        (root / 'STOP').unlink(missing_ok=True)
        (root / 'mim_recovery.cancel').unlink(missing_ok=True)
        launched = time.time()
        with (root / 'mim_recovery.console.log').open('ab', buffering=0) as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '_run',
                    '--core', str(core), '--output', str(root), '--workers', str(args.workers), '--gpu', str(args.gpu)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(150):
            status_path = root / 'status.json'
            current = read(status_path) if status_path.exists() else {}
            if child.poll() is not None or (current.get('pid') == child.pid and current.get('updated_unix', 0) >= launched):
                break
            time.sleep(.1)
        if child.poll() is not None:
            print(('补跑与打包已完成。' if child.returncode == 0 else '启动或补跑未通过，查看：')
                  + str(root / 'mim_recovery.console.log'))
            return child.returncode
    print(f'两项定点补跑已在后台启动，PID={child.pid}；关闭终端可继续运行。')
    print('日志：' + str(root / 'mim_recovery.console.log'))
    print('查看进度：使用相同Python运行 cospec_mim_recover.py watch')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
