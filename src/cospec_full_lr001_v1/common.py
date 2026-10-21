"""Atomic bookkeeping and scientific identities; no torch import in controller."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / 'diagnostics/COSPEC_FULL_LR001_V1'
DEFAULT_REFERENCE = ROOT.parent / 'diagnostics/COSPEC_SUITE_V1'
DATASETS = ('HongHu', 'PaviaU', 'HanChuan', 'LongKou', 'Houston', 'IndianPines', 'Salinas')
EXTERNAL = ('HybridSN', 'A2S2K', 'SQSFormer', 'GSCViT', 'MiM_v2')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hash():
    files = list(ROOT.rglob('*.py')) + [ROOT / n for n in
             ('FROZEN_SHA256.json', 'ORIGINAL_PLANS.json', 'REFERENCE_DATA.json')]
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)})


def verify_frozen():
    for relative, expected in read_json(ROOT / 'FROZEN_SHA256.json').items():
        if file_hash(ROOT / 'frozen' / relative) != expected:
            raise ValueError('冻结模型源码校验失败：' + relative)


def task_key(job):
    return f'patch_{job["patch"]:03d}/{job["dataset"]}/{job["model"]}/seed_{job["seed"]}'


def job_dir(config, job):
    return Path(config['output']) / 'runs' / task_key(job)


def label(job):
    return f'{job["dataset"]}/{job["model"]}/s{job["seed"]}/{job["patch"]}×{job["patch"]}'


def recipe_id(config, job):
    return digest(dict(protocol_id=config['protocol_id'], job=job))


def locked(path):
    path = Path(path)
    if not path.exists():
        return False
    with path.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def identity():
    stat = Path('/proc/self/stat').read_text()
    fields = stat.rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), proc_pid=int(stat.split(' ', 1)[0]), start_ticks=fields[19])


def duration(seconds):
    if seconds is None:
        return '待测'
    minutes = max(0, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f'约{hours}小时{minutes:02d}分' if hours else f'约{minutes}分钟'


def memory_rss_mib():
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.


def make_jobs(windows=(9,)):
    plans = read_json(ROOT / 'ORIGINAL_PLANS.json')
    jobs = [dict(**j, patch=15, group='suite99') for j in plans['COSPEC_SUITE_V1']]
    jobs += [dict(**j, patch=15, group='external105') for j in plans['COSPEC_EXTERNAL_V1']]
    jobs += [dict(dataset='HongHu', model=m, seed=s, patch=p, group='window')
             for p in windows for m in ('B', 'D') for s in (1, 2, 3) if p != 15]
    if len({task_key(j) for j in jobs}) != len(jobs):
        raise ValueError('任务清单重复')
    return jobs


def memory_family(job):
    name = job['model']
    if name in ('A', 'B'):
        return 'linear'
    if name in ('C', 'D', 'D_row', 'D_raster4', 'D_random4'):
        return 'operator'
    return name


def profile_key(config, job):
    d = config['data'][job['dataset']]
    return digest(dict(family=memory_family(job), dataset=job['dataset'], patch=job['patch'],
                       channels=d['shape'][-1], classes=d['classes'], train=config['train'],
                       source=config['source_sha256'], gpu=config['gpu_identity'],
                       backend=config['backend']))[:24]


def batch_count(n, batch):
    count = (n + batch - 1) // batch
    return count - 1 if count > 1 and n % batch == 1 else count


class SafeStop(Exception):
    pass
