#!/usr/bin/env python3
"""Small entry point: background start, viewer, safe stop, and slot cap changes."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,DEFAULT_OUTPUT,read_json,write_json,locked,make_jobs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('start','watch','status','stop','slots'))
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--max-parallel',type=int,default=10)
    p.add_argument('--headless',action='store_true')
    args,extra=p.parse_known_args()
    output=args.output.expanduser().resolve()
    if args.action in ('start','stop','slots') and (output/'config.json').exists():
        if read_json(output/'config.json').get('version')!='cospec_window_lr001_v1':
            raise SystemExit('该目录属于其他实验，未向它写入停止标记、并行设置或启动记录。')
    if not 1<=args.max_parallel<=10:
        raise SystemExit('并行上限只能是1至10')
    if args.action in ('watch','status'):
        from dashboard import watch
        watch(output,once=args.action=='status')
        return 0
    if args.action=='stop':
        if locked(output/'queue.lock') or any(locked(x) for x in output.glob('runs/**/run.lock')):
            (output/'STOP').write_text('save after completed epoch or evaluation batch\n')
            print('已请求保存断点后停止；原命令可续跑。')
        else:
            print('没有运行中的本队列；已保存结果保留。')
        return 0
    if args.action=='slots':
        write_json(output/'runtime_control.json',dict(max_parallel=args.max_parallel))
        print(f'并行上限已设为{args.max_parallel}；正在运行的任务完成后逐步生效。')
        return 0
    from scheduler import parser as build_parser,environment
    command_args=['--output',str(output),'--max-parallel',str(args.max_parallel)]+extra
    parsed=build_parser().parse_args(command_args)
    from window_protocol import protect_output
    protect_output(output,parsed.reference,parsed.parent_output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'launch.lock').open('a+') as launch:
        fcntl.flock(launch,fcntl.LOCK_EX)
        if not locked(output/'queue.lock'):
            (output/'STOP').unlink(missing_ok=True)
            # Do not overwrite the last session's progress count on resume.
            old=read_json(output/'status.json',{})
            status=dict(state='starting',total=len(make_jobs(tuple(parsed.windows))),
                        completed=old.get('completed',0),active=0,active_jobs=[],max_parallel=args.max_parallel,
                        phase='数据与依赖预检',eta_seconds=None,updated_unix=time.time())
            write_json(output/'status.json',status)
            with (output/'controller.log').open('ab',buffering=0) as sink:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scheduler.py')]+command_args,
                      stdout=sink,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=environment(parsed.gpu),
                      cwd=str(ROOT),start_new_session=True)
            # Wait for the actual advisory lock; PID reuse/namespaces do not
            # determine ownership and no process-name based signals are sent.
            for _ in range(100):
                if locked(output/'queue.lock') or child.poll() is not None:
                    break
                time.sleep(.1)
            if child.poll() is not None and child.returncode:
                from dashboard import watch
                watch(output,once=True)
                print('详细错误：'+str(output/'controller.log'))
                return child.returncode
    if not args.headless:
        from dashboard import watch
        watch(output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
