"""Small, atomic experiment bookkeeping; no hidden network operations."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
DATASETS = ('HongHu','PaviaU','HanChuan','LongKou','Houston','IndianPines','Salinas')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def source_hash():
    paths = (list(ROOT.glob('suite_*.py')) + [ROOT/n for n in ('mat_helpers.py','budget_data.py','honghu_data.py')]
             + list((ROOT/'omamba').glob('*.py')) + list((ROOT/'vendor').rglob('*.py')))
    return digest({str(p.relative_to(ROOT)):file_hash(p) for p in sorted(paths)})

def duration(seconds):
    if seconds is None: return '待测'
    seconds = max(0,int(seconds)); h,rest=divmod(seconds,3600); m,s=divmod(rest,60)
    return f'{h}h{m:02d}m{s:02d}s' if h else f'{m}m{s:02d}s'

class Progress:
    def __init__(self,label,total,status_path=None,interval=10):
        self.label,self.total,self.path,self.interval=label,total,status_path,interval
        self.start=self.last=time.monotonic();self.current=0
    def update(self,n,suffix='',force=False):
        self.current=n; now=time.monotonic()
        if force or n==self.total or now-self.last>=self.interval:
            elapsed=now-self.start; eta=elapsed/n*(self.total-n) if n else None
            obj=dict(stage=self.label,completed=n,total=self.total,elapsed_seconds=elapsed,
                     eta_seconds=eta,detail=suffix,updated_unix=time.time(),pid=os.getpid())
            print(f'[{self.label}] {n}/{self.total} ({100*n/max(1,self.total):.1f}%) '
                  f'已用{duration(elapsed)} 剩余≈{duration(eta)} {suffix}',flush=True)
            if self.path:write_json(self.path,obj)
            self.last=now
