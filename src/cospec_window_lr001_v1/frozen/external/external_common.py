"""Bookkeeping and explicit reproducibility settings for the external benchmark."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ('HongHu', 'HanChuan', 'PaviaU', 'LongKou', 'Houston', 'IndianPines', 'Salinas')
MODELS = ('HybridSN', 'A2S2K', 'SQSFormer', 'GSCViT', 'MiM_v2')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def source_hash():
    paths = list(ROOT.glob('*.py')) + list((ROOT / 'vendor').rglob('*.py'))
    paths += [ROOT / 'UPSTREAM.json', ROOT / 'REFERENCE_IDENTITY.json']
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)})

def state_hash(state):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(np.asarray(tuple(value.shape), np.int64).tobytes())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def seed_all(seed):
    import torch
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 before importing torch')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    elif hasattr(torch.backends, 'nnpack'):
        torch.backends.nnpack.set_flags(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Warn-only is recorded; GPU reproducibility is measured by self_check.py,
    # not inferred from this switch or promised for arbitrary CUDA kernels.
    torch.use_deterministic_algorithms(True, warn_only=True)

def capture_rng():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().cpu(),
                cuda=[x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])

def restore_rng(state):
    import torch
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    saved = state.get('cuda', [])
    if bool(saved) != bool(torch.cuda.is_available()):
        raise ValueError('Checkpoint and runtime use different CPU/CUDA RNG regimes')
    if saved:
        if len(saved) != torch.cuda.device_count(): raise ValueError('CUDA device count changed')
        torch.cuda.set_rng_state_all([x.cpu() for x in saved])

def atomic_torch(path, value):
    import torch
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary); os.replace(temporary, path)

def duration(seconds):
    if seconds is None: return '待测'
    seconds = max(0, int(seconds)); hours, rest = divmod(seconds, 3600); minutes, seconds = divmod(rest, 60)
    return f'{hours}h{minutes:02d}m{seconds:02d}s' if hours else f'{minutes}m{seconds:02d}s'

class Progress:
    def __init__(self, label, total, path, interval=10):
        self.label, self.total, self.path = label, total, Path(path)
        self.started = self.last = time.monotonic(); self.interval = interval
    def update(self, completed, detail='', force=False):
        now = time.monotonic()
        if force or completed == self.total or now - self.last >= self.interval:
            elapsed = now - self.started
            eta = elapsed * (self.total - completed) / completed if completed else None
            record = dict(stage=self.label, completed=completed, total=self.total,
                          elapsed_seconds=elapsed, eta_seconds=eta, detail=detail, updated_unix=time.time())
            write_json(self.path, record)
            print(f'[{self.label}] {completed}/{self.total} ({100*completed/max(1,self.total):.1f}%) '
                  f'已用{duration(elapsed)} 剩余≈{duration(eta)} {detail}', flush=True)
            self.last = now
