"""Exactly four compact items. Worker output remains in separate log files."""
import sys
import time
from common import read_json,duration,locked,job_dir,label


def frame(output):
    state=read_json(output/'status.json',{})
    alive=locked(output/'queue.lock')
    total=state.get('total',210)
    completed=state.get('completed',0)
    active=state.get('active_jobs',[])
    failed=state.get('failed',0)
    current=state.get('state','starting')
    finished=current in ('failed','stopped','partial','checked') or (current=='completed' and state.get('phase')=='全部完成')
    if not alive and (current!='starting' or time.time()-state.get('updated_unix',0)>15):
        # A dead controller's last snapshot is not evidence that its workers
        # are still alive. Advisory locks provide that evidence.
        config=read_json(output/'config.json',{})
        active=[dict(label=label(j),phase='running') for j in config.get('jobs',[]) if locked(job_dir(config,j)/'run.lock')]
        if not finished:
            current='interrupted'
    names='；'.join(e['label']+(' [预检]' if e.get('phase')=='显存预检' else '') for e in active) or '无'
    if state.get('error'):
        eta='已停止：'+state['error']
    elif current=='interrupted':
        eta='调度器已退出；原启动命令可接管续跑' if active else '运行已中断；原启动命令可续跑'
    elif current=='completed':
        eta='全部完成' if state.get('phase')=='全部完成' else '汇总打包中'
    elif current=='stopped':
        eta='已安全停止，可按原命令续跑'
    elif current=='partial':
        eta=f'队列结束，失败{failed}项，详见failures.json'
    elif current=='checked':
        eta='预检通过，未启动训练'
    elif current=='stopping':
        eta='正在保存断点后停止'
    elif current=='preflight' or current=='starting':
        eta=state.get('phase','数据与依赖预检')
    elif not alive:
        eta='调度器未运行；已保存断点保留'
    elif state.get('phase')=='等待显存':
        eta='等待可用显存'
    else:
        eta=duration(state.get('eta_seconds'))+'（按实际总吞吐更新）'
    lines=[f'总任务：{total}',f'已完成：{completed}/{total}'+(f'，失败{failed}' if failed else ''),
           f'当前并行：{len(active)}/{state.get("max_parallel",10)}　{names}',f'预计剩余：{eta}']
    return '\n'.join(lines), current, alive


def watch(output,once=False):
    try:
        while True:
            text,state,alive=frame(output)
            if sys.stdout.isatty():
                print('\033[2J\033[H'+text,flush=True)
            else:
                print(text,flush=True)
            if once or state in ('failed','stopped','partial','checked','interrupted') or (state=='completed' and read_json(output/'status.json',{}).get('phase')=='全部完成'):
                return
            time.sleep(10)
    except KeyboardInterrupt:
        # This foreground viewer never sends a stop signal to background jobs.
        return
