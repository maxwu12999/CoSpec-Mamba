"""Resolve and byte-check the unchanged scientific core."""
import hashlib
import json
import os
from pathlib import Path
import sys

ADDON=Path(__file__).resolve().parent
CORE=Path(os.environ.get('COSPEC_CORE',str(ADDON.parent/'cospec_full_lr001_v1'))).expanduser().resolve()


def use_core(check=True):
    if not (CORE/'worker.py').is_file():
        raise FileNotFoundError('缺少原代码包：'+str(CORE)+'；请将两个文件夹并排放在src下')
    sys.path.insert(0,str(CORE))
    if check:
        manifest=json.loads((ADDON/'CORE_IDENTITY.json').read_text())
        for relative,expected in manifest['files'].items():
            if hashlib.sha256((CORE/relative).read_bytes()).hexdigest()!=expected:
                raise ValueError('原科学代码已改变，拒绝混用断点：'+relative)
        import common
        if common.source_hash()!=manifest['source_sha256']:
            raise ValueError('原代码包存在额外或修改的Python文件；请保留原审计版代码')
    return CORE


def revision():
    files={str(p.relative_to(ADDON)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(ADDON.glob('*')) if p.suffix in ('.py','.json','.sh')}
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
