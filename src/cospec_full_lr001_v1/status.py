#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys
if __name__=='__main__':
    raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).with_name('control.py')),'status']+sys.argv[1:]))
