#!/usr/bin/env python3
"""Synthetic checks only: never formal accuracy or proof of CUDA bitwise replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import contextlib
import io
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from common import (ROOT, SafeStop, digest, file_hash, job_dir, make_jobs, read_json,
                    recipe_id, source_hash, task_key, write_json)
from data import Patches, all_cases, mask_hash
from models import build_model, configure_imports
configure_imports()
from external_common import seed_all, state_hash
import worker
import speed_data
import speed_eval
from policy import ThroughputETA, can_admit, requires_exclusive
from window_protocol import protect_output, verify_science


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(32, 12), nn.BatchNorm1d(12), nn.SiLU(), nn.Dropout(.2))
        self.head = nn.Linear(12, 3)

    def forward(self, x, observed):
        clean = torch.where(observed[:, :, None, None], x, torch.zeros_like(x)).mean((-1, -2))
        feature = self.features(clean)
        return dict(logits=self.head(feature), embedding=feature)


def fixture(root):
    cache = Path(root) / 'cache'
    cache.mkdir(parents=True)
    cube = np.random.default_rng(42).normal(size=(16, 16, 32)).astype(np.float32)
    tr, te = np.zeros((16, 16), np.int64), np.zeros((16, 16), np.int64)
    tr.ravel()[:22] = np.arange(22) % 3 + 1
    te.ravel()[50:80] = np.tile(np.arange(1, 4), 10)
    for name, value in (('cube', cube), ('tr', tr), ('te', te), ('wave', np.linspace(400, 1000, 32))):
        np.save(cache / (name + '.npy'), value)
    np.savez(cache / 'stats.npz', means=np.stack([cube[tr == c].mean(0) for c in (1, 2, 3)]),
             variances=np.ones((3, 32), np.float32))
    cases = [dict(name='full', fraction=1., repeat=-1, mask=[True] * 32, sha256=mask_hash([True] * 32))]
    ordering = np.random.default_rng(20261014)
    for repeat in (1, 2, 3):
        order = ordering.permutation(32)
        for fraction in (.8, .6, .4):
            mask = np.zeros(32, bool)
            mask[order[:round(32 * fraction)]] = True
            cases.append(dict(name=f'keep{round(fraction * 100)}_r{repeat}', fraction=fraction,
                              repeat=repeat, mask=mask.tolist(), sha256=mask_hash(mask)))
    meta = dict(dataset='HongHu', shape=[16, 16, 32], classes=3, train_count=22, test_count=30,
                class_support=[10, 10, 10], cache=str(cache), cases=all_cases(dict(shape=[16, 16, 32], cases=cases)))
    return dict(output=str(Path(root) / 'output'), protocol_id='synthetic_pipeline_only', scope='pipeline_check',
                source_sha256='synthetic_fixture_only', device='cpu', data={'HongHu': meta}, pca={},
                backend=dict(suite='reference', external='native'), runtime_speedup=dict(prefetch=True),
                gpu_identity=dict(name='CPU fixture', total_mib=24564.),
                train=dict(epochs=100, batch=7, eval_batch=8, lr=.001, lr_policy='constant'))


def compare_metrics(left, right):
    ignored = {'seconds_this_segment', 'timing_scope'}
    assert len(left) == len(right) == 16
    for a, b in zip(left, right):
        for key in set(a) - ignored:
            if a[key] is None:
                assert b[key] is None
            elif isinstance(a[key], (float, int, list)):
                np.testing.assert_allclose(a[key], b[key], rtol=0, atol=0, err_msg=key)
            else:
                assert a[key] == b[key], key


def compare_arrays(left, right, cases):
    for case in cases:
        with np.load(left / (case['name'] + '.npz')) as a, np.load(right / (case['name'] + '.npz')) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=case['name'] + ':' + key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    original_source = source_hash()
    verify_science()
    jobs = make_jobs()
    assert len(jobs) == len(set(map(task_key, jobs))) == 12
    assert {j['patch'] for j in jobs} == {7, 11}
    assert {j['dataset'] for j in jobs} == {'HongHu'}
    assert {j['model'] for j in jobs} == {'B', 'D'}
    assert all(sum(j['patch'] == p and j['model'] == m for j in jobs) == 3
               for p in (7, 11) for m in ('B', 'D'))
    expected = read_json(ROOT / 'UPSTREAM_SCIENCE.json')
    assert expected['train'] == dict(epochs=100, batch=32, eval_batch=128, lr=.001, lr_policy='constant',
                                    optimizer='Adam', weight_decay=0., clip=15., two_view_ce=True, min_keep=.6, max_keep=.9)
    actual_cases = all_cases(read_json(ROOT / 'REFERENCE_DATA.json')['HongHu'])
    assert [c['sha256'] for c in actual_cases] == expected['data_identity']['HongHu']['cases']
    for repeat in (1, 2, 3):
        masks = [np.array(next(c['mask'] for c in actual_cases if c['name'] == f'keep{percent}_r{repeat}'))
                 for percent in (10, 20, 40, 60, 80)]
        assert all(np.all(a <= b) for a, b in zip(masks, masks[1:]))
        assert masks[0].sum() == 27 and masks[1].sum() == 54
    print('[通过] 仅12项新训练；原训练/模型/加速源文件逐字节一致；16掩码与原归档一致。', flush=True)
    profile = dict(train_peak_reserved_mib=4000., eval_peak_reserved_mib=7000., rss_peak_mib=500.)
    assert can_admit(profile, [], {'p': profile}, dict(free_mib=24000.))
    assert not can_admit(profile, [], {'p': profile}, dict(free_mib=7000.))
    assert not can_admit(profile, [], {'p': profile}, dict(free_mib=24000.), available_ram=500.)
    assert not can_admit(profile, [dict(profile_key='p', actual_gpu_mib=0.)] * 2,
                         {'p': profile}, dict(free_mib=24000.))
    assert requires_exclusive(profile, 3., 1024., 24564.)
    eta = ThroughputETA()
    assert eta.update(0., 0., 100.) is None and eta.update(10., 5., 95.) == 190.
    with tempfile.TemporaryDirectory(prefix='cospec_windows_check_') as temporary:
        root = Path(temporary)
        cfg = fixture(root)
        meta = cfg['data']['HongHu']
        for protected in (ROOT, root / 'parent', root / 'reference'):
            try:
                protect_output(protected, root / 'reference', root / 'parent')
            except ValueError:
                pass
            else:
                raise AssertionError('没有阻止覆盖旧目录')
        original_batch = Patches.batch
        for patch_size in (7, 11):
            for seed in (None, 1, 19):
                old, new = Patches(meta, 'tr', patch_size, seed), Patches(meta, 'tr', patch_size, seed)
                for indices in ([0], [21, 0, 15, 3, 20], list(range(13))):
                    assert all(torch.equal(a, b) for a, b in
                               zip(original_batch(old, indices), speed_data.vector_batch(new, indices)))
                    if seed is not None:
                        assert digest(old.rng.bit_generator.state) == digest(new.rng.bit_generator.state)
        print('[通过] 7/11窗口批量加速：边缘像元、随机旋转、奇数batch、私有随机状态与原读取一致。', flush=True)
        for patch_size in (7, 11):
            job = dict(dataset='HongHu', model='D', seed=2, patch=patch_size, group='synthetic')
            cfg['jobs'] = [job]
            def training(name, fast, stop_epoch=None, checkpoint=None):
                speed_data.uninstall()
                if fast:
                    speed_data.install(prefetch=True)
                folder = root / f'{patch_size}_{name}'
                folder.mkdir()
                if checkpoint:
                    import shutil
                    shutil.copy2(checkpoint, folder / 'last.pt')
                seed_all(job['seed'])
                model = TinyNet()
                progress = worker.Progress(folder / 'progress.json', recipe_id(cfg, job))
                try:
                    worker.train(model, cfg, job, folder, progress, lambda: False, stop_after_epoch=stop_epoch)
                except SafeStop:
                    assert stop_epoch is not None
                return folder, model
            with contextlib.redirect_stdout(io.StringIO()):
                old_folder, old = training('legacy', False)
                fast_folder, fast = training('fast', True)
                stopped, _ = training('stopped', True, stop_epoch=73)
                resumed_folder, resumed = training('resumed', True, checkpoint=stopped / 'last.pt')
            assert state_hash(old.state_dict()) == state_hash(fast.state_dict()) == state_hash(resumed.state_dict())
            saved = [torch.load(p / 'last.pt', map_location='cpu', weights_only=False)
                     for p in (old_folder, fast_folder, resumed_folder)]
            for s in saved:
                assert s['epoch'] == len(s['history']) == 100
                assert all(v['lr'] == .001 for v in s['history'].values())
                assert digest(s['rotation_rng']) == digest(saved[0]['rotation_rng'])
                assert digest(s['mask_rng']) == digest(saved[0]['mask_rng'])
                assert torch.equal(s['loader_rng'], saved[0]['loader_rng'])
                assert torch.equal(s['rng']['torch'], saved[0]['rng']['torch'])
            speed_data.uninstall()
            sha = state_hash(old.state_dict())
            baseline = worker.evaluate(old, cfg, job, old_folder, sha,
                                       worker.Progress(old_folder / 'progress.json', recipe_id(cfg, job)), lambda: False)
            speed_data.install(prefetch=True)
            rapid = speed_eval.evaluate(fast, cfg, job, fast_folder, sha,
                                        worker.Progress(fast_folder / 'progress.json', recipe_id(cfg, job)), lambda: False)
            compare_metrics(baseline, rapid)
            compare_arrays(old_folder, fast_folder, meta['cases'])
            progress = worker.Progress(resumed_folder / 'progress.json', recipe_id(cfg, job))
            try:
                speed_eval.evaluate(resumed, cfg, job, resumed_folder, sha, progress,
                                    lambda: progress.values.get('test_steps_done', 0) >= 7)
            except SafeStop:
                pass
            else:
                raise AssertionError('未在测试中断')
            continued = speed_eval.evaluate(resumed, cfg, job, resumed_folder, sha, progress, lambda: False)
            compare_metrics(baseline, continued)
            compare_arrays(old_folder, resumed_folder, meta['cases'])
            print(f'[通过] {patch_size}×{patch_size}：100轮固定0.001及73轮断点恢复；16条件测试/恢复与原代码一致。', flush=True)
        speed_data.uninstall()
        if args.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('当前CUDA不可用')
            cfg['device'] = 'cuda'
            cfg['backend']['suite'] = 'cuda'
        for patch_size in (7, 11):
            x, y = Patches(meta, 'tr', patch_size).batch(range(2))
            x, y = x.to(cfg['device']), y.to(cfg['device'])
            for model_name in ('B', 'D'):
                seed_all(2)
                job = dict(dataset='HongHu', model=model_name, seed=2, patch=patch_size, group='synthetic')
                model = build_model(cfg, job).to(cfg['device'])
                mask = torch.ones((2, 32), dtype=torch.bool, device=cfg['device'])
                mask[:, 3:] = False
                out = model(x, mask)['logits']
                assert out.shape == (2, 3) and torch.isfinite(out).all()
                torch.nn.functional.cross_entropy(out, y).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 15., error_if_nonfinite=True)
                model.eval()
                with torch.inference_mode():
                    before = model(x, torch.ones_like(mask))['logits'].clone()
                    model(x, mask)
                    after = model(x, torch.ones_like(mask))['logits']
                assert torch.equal(before, after)
                del model, out
                print(f'[通过] 实际B/D模型前后向及掩码切换：{patch_size}×{patch_size}/{model_name}。', flush=True)
    assert source_hash() == original_source
    print('全部自检通过。合成CPU结果不属于正式实验，也不代表CUDA逐位可复现。', flush=True)


if __name__ == '__main__':
    main()
