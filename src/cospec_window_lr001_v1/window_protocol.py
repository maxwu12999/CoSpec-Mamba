"""Read-only comparison to the archived fixed-LR windows; new, isolated runs."""
import importlib.metadata
import math
from pathlib import Path
import sys
import time

from common import (ROOT, digest, file_hash, job_dir, locked, make_jobs,
                    read_json, source_hash, task_key, verify_frozen, write_json)

VERSION = 'cospec_window_lr001_v1'


def overlaps(a, b):
    a, b = Path(a).expanduser().resolve(), Path(b).expanduser().resolve()
    return a == b or a in b.parents or b in a.parents


def protect_output(output, reference, parent_output):
    root = Path(output).expanduser().resolve()
    for protected in (ROOT, reference, parent_output):
        if overlaps(root, protected):
            raise ValueError('补跑输出必须独立于原210项、原数据缓存和代码目录：' + str(root))
    if (root / 'config.json').exists():
        if read_json(root / 'config.json').get('version') != VERSION:
            raise ValueError('该目录属于其他实验；没有写入或覆盖：' + str(root))


def verify_science():
    manifest = read_json(ROOT / 'UPSTREAM_SCIENCE.json')
    verify_frozen()
    for name, expected in {**manifest['unchanged_files'], **manifest['speedup_files']}.items():
        if file_hash(ROOT / name) != expected:
            raise ValueError('原训练/模型/读取/加速代码被改变：' + name)
    return manifest


def parent_jobs(parent):
    chosen = []
    for patch in (9, 15):
        for model in ('B', 'D'):
            for seed in (1, 2, 3):
                found = [j for j in parent['jobs'] if
                         (j['dataset'], j['model'], j['seed'], j['patch']) == ('HongHu', model, seed, patch)]
                if len(found) != 1:
                    raise ValueError(f'原9×9/15×15清单缺失或重复：{model}/s{seed}/{patch}')
                chosen.extend(found)
    return chosen


def validate_parent(parent_root, manifest):
    from results import complete
    parent_root = Path(parent_root).expanduser().resolve()
    parent = read_json(parent_root / 'config.json')
    for key, expected in (
        ('version', 'cospec_full_lr001_v1'), ('scope', 'formal'),
        ('source_sha256', manifest['parent_source_sha256']),
        ('protocol_id', manifest['parent_protocol_id']),
        ('train', manifest['train']), ('backend', manifest['backend'])):
        if parent.get(key) != expected:
            raise ValueError('原9×9/15×15协议与固定0.001归档不符：' + key)
    if parent['data_identity']['HongHu'] != manifest['data_identity']['HongHu']:
        raise ValueError('原HongHu数据身份/测试掩码与归档不符')
    from data import mask_hash
    meta = parent['data']['HongHu']
    actual_identity = dict(data_id=meta['data_id'], cache_sha256=meta['cache_sha256'],
                           cases=[c['sha256'] for c in meta['cases']])
    if actual_identity != manifest['data_identity']['HongHu'] or any(
            mask_hash(c['mask']) != c['sha256'] for c in meta['cases']):
        raise ValueError('原HongHu的实际掩码内容/顺序与已声明哈希不一致')
    # Bind results to their actual disk location without changing their scientific ID.
    parent = dict(parent, output=str(parent_root))
    chosen = parent_jobs(parent)
    if locked(parent_root / 'queue.lock') or any(locked(job_dir(parent, j) / 'run.lock') for j in chosen):
        raise RuntimeError('原210项队列仍在运行；请待其完成后启动本补跑。未向原进程发送信号。')
    records = []
    for job in chosen:
        if not complete(parent, job):
            raise ValueError('原9×9/15×15任务尚未完整完成：' + task_key(job))
        directory = job_dir(parent, job)
        history = read_json(directory / 'history.json')
        if set(history) != {str(i) for i in range(1, 101)}:
            raise ValueError('原训练记录没有完整覆盖100轮：' + task_key(job))
        for epoch in range(1, 101):
            row = history[str(epoch)]
            if row['epoch'] != epoch or row['lr'] != .001:
                raise ValueError('原结果包含分段学习率：' + task_key(job))
            if any(not math.isfinite(row[k]) for k in ('loss', 'full_ce', 'missing_ce')):
                raise ValueError('原训练记录含非有限损失：' + task_key(job))
            if row['training_centers_seen'] != parent['data']['HongHu']['train_count']:
                raise ValueError('原训练没有完整覆盖训练中心：' + task_key(job))
        result_path = directory / 'result.json'
        records.append(dict(job=job, result_sha256=file_hash(result_path),
                            history_sha256=file_hash(directory / 'history.json')))
    return parent, records


def snapshot_parent(parent, records, output):
    snapshot = Path(output) / 'parent_snapshot'
    # These are only JSON copies; original weights/results are never loaded or written.
    write_json(snapshot / 'config.json', parent)
    for record in records:
        original = job_dir(parent, record['job'])
        target = snapshot / 'runs' / task_key(record['job'])
        for name, checksum in (('result.json', 'result_sha256'), ('history.json', 'history_sha256')):
            value = read_json(original / name)
            if file_hash(original / name) != record[checksum]:
                raise ValueError('读取期间原结果发生改变：' + str(original / name))
            write_json(target / name, value)
    write_json(snapshot / 'provenance.json', dict(
        original_output=parent['output'], original_protocol_id=parent['protocol_id'],
        original_config_sha256=file_hash(Path(parent['output']) / 'config.json'),
        records=records, imported_windows=[9, 15], copied_checkpoints=False,
        scope='Read-only JSON copies for comparison, not new training tasks.'))


def prepare_config(args, hardware):
    if not 1 <= args.max_parallel <= 10 or args.eval_batch != 128 or args.headroom_mib < 512:
        raise ValueError('并行上限1至10；测试batch固定128；显存余量至少512MiB')
    protect_output(args.output, args.reference, args.parent_output)
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = verify_science()
    parent, records = validate_parent(args.parent_output, manifest)
    from data import validate_caches
    data = validate_caches(args.reference.expanduser(), root, datasets=('HongHu',))
    meta, old_meta = data['HongHu'], parent['data']['HongHu']
    for key in ('dataset', 'shape', 'classes', 'train_count', 'test_count', 'data_id',
                'cache_sha256', 'normalization', 'class_support', 'cases'):
        if meta[key] != old_meta[key]:
            raise ValueError('补跑与9×9/15×15缓存/掩码不一致：' + key)
    source = source_hash()
    protocol = dict(version=VERSION, source_sha256=source, scope='formal', jobs=make_jobs(),
                    train=dict(manifest['train']), backend=dict(manifest['backend']),
                    data_identity={'HongHu': manifest['data_identity']['HongHu']}, pca_identity={},
                    parent_protocol_id=parent['protocol_id'],
                    scientific_core_sha256=manifest['parent_source_sha256'],
                    runtime_speedup=dict(enabled=True, prefetch=True,
                        archived_revision=manifest['archived_speedup_revision'],
                        speedup_files=manifest['speedup_files']))
    pid = digest(protocol)
    gpu_identity = {k: hardware[k] for k in ('name', 'uuid', 'total_mib', 'driver')}
    gpu_identity['torch'] = importlib.metadata.version('torch')
    config = dict(**protocol, protocol_id=pid, output=str(root),
                  reference=str(args.reference.expanduser().resolve()),
                  parent_output=str(args.parent_output.expanduser().resolve()),
                  data=data, pca={}, device='cuda', gpu_identity=gpu_identity,
                  runtime_source_sha256=source)
    path = root / 'config.json'
    old = read_json(path) if path.exists() else None
    if old is not None:
        if old['protocol_id'] != pid:
            raise ValueError('已有补跑目录协议不同；请选择新目录。没有覆盖已保存的权重/结果。')
        if old != config:
            write_json(root / ('configuration_backup_' + time.strftime('%Y%m%d_%H%M%S') + '.json'), old)
    snapshot_parent(parent, records, root)
    write_json(path, config)
    write_json(root / 'runtime_control.json', dict(max_parallel=args.max_parallel))
    for folder in ('logs', 'profiles'):
        (root / folder).mkdir(exist_ok=True)
    write_json(root / 'environment.json', dict(
        gpu=gpu_identity, python=sys.version, protocol_id=pid,
        original_scientific_core=manifest['parent_source_sha256'],
        runtime_source_sha256=source, learning_rate='constant_0.001',
        upstream_files_verified=True, old_results_read_only=True))
    write_json(root / 'protocol_audit.json', dict(
        passed=True, jobs=12, dataset='HongHu', models=['B', 'D'], windows=[7, 11], seeds=[1, 2, 3],
        train=config['train'], shared_mask_count=16, parent_windows_verified=[9, 15],
        parent_completed_results=len(records), parent_epochs_checked=1200,
        cache_identity=protocol['data_identity'], model_options=manifest['model_options'],
        training_and_model_sources_unchanged=True, new_checkpoints=True,
        originals_written=False))
    return config
