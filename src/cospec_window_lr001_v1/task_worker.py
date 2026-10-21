#!/usr/bin/env python3
"""Thin runtime wrapper; the archived training/optimizer code stays byte-exact."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys


def install():
    import worker
    import speed_data
    import speed_eval
    speed_data.install(prefetch=True)
    worker.evaluate = speed_eval.evaluate


def main():
    from common import read_json, source_hash
    from window_protocol import verify_science
    verify_science()
    path = Path(sys.argv[sys.argv.index('--config') + 1])
    config = read_json(path)
    if config['source_sha256'] != source_hash():
        raise ValueError('启动后源码改变，拒绝混用本次检查点')
    if config['train']['lr'] != .001 or config['train']['lr_policy'] != 'constant':
        raise ValueError('本补跑只允许全程固定0.001')
    install()
    import worker
    return worker.main()


if __name__ == '__main__':
    raise SystemExit(main())
