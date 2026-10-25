#!/usr/bin/env python3
"""Fixed-protocol component ablation, with the archived train/eval implementation."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import zipfile
import bootstrap
import numpy as np
import torch
from common import (read_json, write_json, file_hash, digest, recipe_id, job_dir, SafeStop)
from data import validate_caches
from worker import train, evaluate, Progress
from external_common import seed_all, state_hash
from internal_models import build, VARIANTS
from summarize import summarize

ROOT = bootstrap.ROOT
DEFAULT_OUTPUT = ROOT.parent / 'diagnostics/COSPEC_INTERNAL_V1'


def code_identity():
    frozen = read_json(ROOT / 'ARCHIVED_SHA256.json')
    for name, expected in frozen.items():
        if file_hash(bootstrap.VENDOR / name) != expected:
            raise ValueError('Archived scientific source changed: ' + name)
    paths = list(ROOT.rglob('*.py')) + [ROOT / 'ARCHIVED_SHA256.json']
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)})


def default_reference():
    candidates = [ROOT.parent / 'diagnostics/COSPEC_SUITE_V1',
                  Path.cwd() / 'diagnostics/COSPEC_SUITE_V1',
                  Path.cwd() / 'src/diagnostics/COSPEC_SUITE_V1']
    return next((p for p in candidates if (p/'cache/HongHu/data.json').exists()), candidates[0])


def prepare(args):
    output = args.output.resolve()
    if args.synthetic and not output.name.endswith('_SYNTHETIC'):
        raise ValueError('Synthetic check output must end in _SYNTHETIC; never mix it with formal results')
    reference = (args.reference or default_reference()).resolve()
    print('校验原划分、归一化、波长、文件哈希及共享测试掩码：', reference, flush=True)
    data = validate_caches(reference, output, tuple(args.datasets), synthetic=args.synthetic)
    if not args.synthetic and (not args.device.startswith('cuda') or not torch.cuda.is_available()):
        raise ValueError('正式运行要求原CUDA环境；CPU仅用于显式的synthetic自检，不会静默替换Mamba')
    if not 0 < args.svd_rtol < 1:
        raise ValueError('SVD rtol must lie between zero and one')
    backend = 'reference' if args.device == 'cpu' else 'cuda'
    if backend == 'cuda':
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    architecture = dict(dim=8 if args.synthetic else 64, depth=4,
                        basis=8 if args.synthetic else 64, patch=3 if args.synthetic else 15)
    training = dict(epochs=2 if args.synthetic else 100, batch=4 if args.synthetic else 32,
                    eval_batch=4 if args.synthetic else 128, lr=.001, weight_decay=0., clip=15.,
                    rotation_degrees=[0,90,180,270], dual_view_ce_weights=[.5,.5],
                    training_keep_range=[.6,.9], endpoint='last_epoch', amp=False)
    # backend / library / GPU identities protect continuation of stochastic runs.
    runtime = dict(torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                   backend=backend, device=args.device,
                   gpu=torch.cuda.get_device_name(torch.device(args.device)) if backend=='cuda' else None)
    science = dict(version='COSPEC_INTERNAL_V1_WORD_V16', scope='synthetic' if args.synthetic else 'formal',
        architecture=architecture, train=training, svd_rtol=args.svd_rtol, svd_atol=0.,
        svd_dtype='float32', svd_method='torch.linalg.svd full_matrices=False; strict s>rtol*smax',
        ridge=.01, correction_bound=.1, refiner_initialization='zeros',
        test_mask_seed=20261014, retained_count='archived round(), exact masks hash-checked',
        datasets={d:{k:v for k,v in m.items() if k!='cache'} for d,m in data.items()},
        runtime=runtime, source_sha256=code_identity())
    jobs = [dict(dataset=d, model=v, seed=s, patch=architecture['patch'], group='internal_components')
            for d in args.datasets for s in args.seeds for v in args.variants]
    config = dict(protocol_id=digest(science), science=science, source_sha256=science['source_sha256'],
        scope=science['scope'], reference=str(reference), output=str(output), data=data,
        datasets=args.datasets, seeds=args.seeds, variants=args.variants, jobs=jobs, train=training,
        architecture=architecture, device=args.device, backend=dict(suite=backend), svd_rtol=args.svd_rtol)
    old = output / 'config.json'
    if old.exists():
        saved = read_json(old)
        if saved['protocol_id'] != config['protocol_id'] or saved['jobs'] != jobs:
            raise ValueError('该输出目录已有不同协议/任务，拒绝覆盖；换用新的--output目录')
    write_json(old, config)
    return config


def model_for(config, job):
    cache = Path(config['data'][job['dataset']]['cache'])
    seed_all(job['seed'])
    return build(job['model'], np.load(cache/'wave.npy'), dict(np.load(cache/'stats.npz')),
                 job['seed'], backend=config['backend']['suite'], svd_rtol=config['svd_rtol'],
                 **config['architecture'])


def audit_initialization(config, job, model):
    group_file = Path(config['output']) / 'initialization_groups' / f'{job["dataset"]}_seed{job["seed"]}.json'
    group = read_json(group_file, {})
    common = state_hash(model.backbone_state())
    if group and group['common_backbone_sha256'] != common:
        raise ValueError('同种子公共主干/位置/读出初始化不一致')
    projection = getattr(model.operator, 'projection', None)
    proj_sha = state_hash(projection.state_dict()) if projection is not None else None
    if proj_sha and group.get('fixed_basis_projection_sha256', proj_sha) != proj_sha:
        raise ValueError('固定基各组的公共投影初始化不一致')
    group.update(common_backbone_sha256=common)
    if proj_sha:
        group['fixed_basis_projection_sha256'] = proj_sha
    init = dict(recipe_id=recipe_id(config, job), state_sha256=state_hash(model.state_dict()),
                common_backbone_sha256=common, feature_projection_sha256=proj_sha,
                counts=model.component_counts(), variant=job['model'])
    directory = job_dir(config, job); directory.mkdir(parents=True, exist_ok=True)
    old = directory/'initialization.json'
    if old.exists() and read_json(old) != init:
        raise ValueError('初始化改变，拒绝复用该检查点')
    write_json(old, init)
    group.setdefault('variants', {})[job['model']] = init
    write_json(group_file, group)
    return init


class ConsoleProgress(Progress):
    def __init__(self, path, rid, label):
        super().__init__(path, rid)
        self.label, self.console_last, self.phase = label, 0., None

    def update(self, force=False, **values):
        super().update(force=force, **values)
        now = time.monotonic()
        phase = self.values.get('phase')
        if force or phase != self.phase or now-self.console_last >= 10:
            s = self.values
            if phase == 'train':
                done, total = s.get('train_steps_done',0), s.get('train_steps_total',1)
                detail = f'epoch={s.get("epoch",0)} batch={s.get("batch",0)}/{s.get("steps_per_epoch",0)}'
            else:
                done, total = s.get('test_steps_done',0), s.get('test_steps_total',1)
                detail = s.get('case', phase or '')
            print(f'[{self.label}] {phase}: {done}/{total} ({100*done/max(1,total):.1f}%) {detail}', flush=True)
            self.console_last, self.phase = now, phase


def run_one(config, job, stop, stop_after_epoch=None):
    directory = job_dir(config, job); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rid = recipe_id(config, job)
        if (directory/'result.json').exists():
            old = read_json(directory/'result.json')
            if old['recipe_id'] != rid or not old.get('complete'):
                raise ValueError('完成结果不匹配，拒绝跳过')
            return
        (directory/'outcome.json').unlink(missing_ok=True)
        label = f'{job["dataset"]}/{job["model"]}/seed{job["seed"]}'
        progress = ConsoleProgress(directory/'progress.json', rid, label)
        model = model_for(config, job)
        initial = audit_initialization(config, job, model)
        model = model.to(config['device'])
        history = train(model, config, job, directory, progress, stop, stop_after_epoch=stop_after_epoch)
        if sorted(map(int, history)) != list(range(1, config['train']['epochs']+1)):
            raise ValueError('训练历史缺轮次，不输出正式结果')
        sha = state_hash(model.state_dict())
        metrics = evaluate(model, config, job, directory, sha, progress, stop)
        write_json(directory/'result.json', dict(complete=True, recipe_id=rid, job=job,
            scope=config['scope'], completed_epochs=config['train']['epochs'], metrics=metrics,
            model_sha256=sha, initial_state_sha256=initial['state_sha256'],
            source_sha256=config['source_sha256'], lr_policy='constant_0.001',
            train_seconds=sum(x['seconds'] for x in history.values())))
        progress.update(force=True, phase='completed')
        del model
        gc.collect()
        if config['device'].startswith('cuda'):
            torch.cuda.empty_cache()


def export(output):
    output = Path(output).resolve()
    summarize(output)
    dest = output.parent / (output.name + '_results.zip')
    temporary = dest.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.csv','.md','.log') and 'partial' not in path.parts:
                z.write(path, str(Path(output.name)/path.relative_to(output)))
    temporary.replace(dest)
    print('结果包（不含影像和权重）：', dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('run', help='准备/运行/自动续跑；默认30项独立训练')
    p.add_argument('--reference', type=Path)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--datasets', nargs='+', choices=('HongHu','PaviaU'), default=['HongHu','PaviaU'])
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--seeds', nargs='+', type=int, choices=(1,2,3), default=[1,2,3])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--svd-rtol', type=float, default=1e-6)
    p.add_argument('--check-only', action='store_true', help='校验全部数据与公共初始化，不训练')
    p.add_argument('--synthetic', action='store_true', help='仅自检：2轮/3×3/8维，不是论文实验')
    for name in ('summary','export','stop'):
        p = commands.add_parser(name); p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == 'summary':
        print('完成数/总数：', summarize(args.output)); return 0
    if args.command == 'export':
        export(args.output); return 0
    if args.command == 'stop':
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'STOP').touch(); print('已请求安全停止：训练在本轮保存后退出，测试在批次保存后退出'); return 0
    if any(len(xs)!=len(set(xs)) for xs in (args.datasets,args.seeds,args.variants)):
        parser.error('数据集、种子、变体不得重复')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'queue.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(args)
        torch.set_num_threads(1)
        if args.check_only:
            for job in config['jobs']:
                model = model_for(config,job); audit_initialization(config,job,model); del model
            summarize(args.output)
            print('检查通过；正式训练尚未开始。'); return 0
        (args.output/'STOP').unlink(missing_ok=True)
        stopped = [False]
        signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0,True))
        signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0,True))
        stop = lambda: stopped[0] or (args.output/'STOP').exists()
        for index, job in enumerate(config['jobs'],1):
            if stop():
                return 130
            print(f'\n[{index}/{len(config["jobs"])}] {job}', flush=True)
            try:
                run_one(config,job,stop)
            except SafeStop:
                write_json(job_dir(config,job)/'outcome.json', dict(state='stopped', recipe_id=recipe_id(config,job)))
                summarize(args.output)
                print('断点已保存；重新执行相同命令即可续跑。'); return 130
            except Exception as error:
                write_json(job_dir(config,job)/'outcome.json', dict(state='failed', error=str(error), traceback=traceback.format_exc()))
                raise
            done,total=summarize(args.output)
            print(f'队列完成 {done}/{total}；汇总已更新。', flush=True)
        export(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

