#!/usr/bin/env python3
from bootstrap import use_core
use_core()
import sys
from pathlib import Path
import worker as core_worker
from common import read_json
from speed_runtime import install

if __name__=='__main__':
    config_path=Path(sys.argv[sys.argv.index('--config')+1])
    config=read_json(config_path)
    runtime=config.get('runtime_speedup',{})
    if runtime.get('enabled',True):install(prefetch=runtime.get('prefetch',True))
    raise SystemExit(core_worker.main())
