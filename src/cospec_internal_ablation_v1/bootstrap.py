"""Use the bundled, unchanged experiment core; never import a nearby old runner."""
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor'
for path in (VENDOR, VENDOR / 'frozen/suite', VENDOR / 'frozen/external'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

