#!/usr/bin/env python3
"""Save old workers, wait for real locks, then launch the background fast queue."""
from bootstrap import ADDON,CORE,use_core,revision
use_core()
import fcntl
import os
from pathlib import Path
import sys
from common import read_json,write_json,locked,job_dir
import control
import scheduler


def main():
    original='--original' in sys.argv
    if original:sys.argv.remove('--original')
    headless='--headless' in sys.argv
    if headless:sys.argv.remove('--headless')
    args=scheduler.parser().parse_args()
    root=args.output.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'speedup_launch.lock').open('a+') as switch:
        try:fcntl.flock(switch,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('已有切换/查看进程；可用原watch.sh查看进度')
        old=read_json(root/'config.json',{})
        same=not original and old.get('runtime_speedup',{}).get('revision')==revision()
        if not (same and locked(root/'queue.lock')):
            if locked(root/'queue.lock') or any(locked(p) for p in root.glob('runs/**/run.lock')):
                print('正在保存当前轮/测试批次；随后从同一断点切换运行入口……',flush=True)
                (root/'STOP').write_text('save before runtime speedup switch\n')
            with (root/'queue.lock').open('a+') as queue:
                fcntl.flock(queue,fcntl.LOCK_EX)
                # Covers an old controller that exited before its workers did.
                for job in old.get('jobs',[]):
                    path=job_dir(old,job)/'run.lock'
                    if path.exists():
                        with path.open('a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX)
            (root/'STOP').unlink(missing_ok=True)
        control.ROOT=CORE if original else ADDON
        sys.argv=[str(ADDON/'launcher.py'),'start','--headless']+sys.argv[1:]
        code=control.main()
    # Keep the switch lock only during launch, so another terminal can roll back.
    if not code and not headless:
        from dashboard import watch
        watch(root)
    return code


if __name__=='__main__':raise SystemExit(main())
