#!/usr/bin/env python3
"""Memory-peak reservations, at most ten independent GPU workers, measured ETA."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import fcntl
import importlib.metadata
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from common import (ROOT, DEFAULT_OUTPUT, DEFAULT_REFERENCE, DATASETS, read_json, write_json,
                    make_jobs, job_dir, recipe_id, profile_key, source_hash, verify_frozen,
                    identity, locked, digest, label, task_key, SafeStop)
from policy import can_admit, estimated_work, work_done, ThroughputETA, requires_exclusive, reservation

RAMP_SECONDS=2.
POLL_SECONDS=.5


def gpu_info(index):
    p=subprocess.run(['nvidia-smi','-i',str(index),'--query-gpu=name,uuid,memory.total,memory.free,driver_version',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8,check=True)
    name,uuid,total,free,driver=[v.strip() for v in p.stdout.strip().splitlines()[0].split(',')]
    return dict(name=name,uuid=uuid,total_mib=float(total),free_mib=float(free),driver=driver)


def gpu_processes(index):
    try:
        p=subprocess.run(['nvidia-smi','-i',str(index),'--query-compute-apps=pid,used_gpu_memory',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        result={}
        for line in p.stdout.splitlines():
            pid,memory=[x.strip() for x in line.split(',')]
            result[int(pid)]=float(memory)
        return result
    except (OSError,ValueError,subprocess.SubprocessError):
        return {}


def available_ram():
    limits=[]
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                limits.append(int(line.split()[1])/1024)
    except OSError:
        pass
    # AutoDL containers can have much less RAM than the host reported above.
    for limit_name,current_name in (
        ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
        ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')):
        try:
            limit=Path(limit_name).read_text().strip()
            if limit!='max':
                maximum=int(limit)
                if maximum<2**60:
                    current=int(Path(current_name).read_text().strip())
                    limits.append(max(0,maximum-current)/2**20)
        except (OSError,ValueError):
            pass
    return min(limits) if limits else None


def load_profile(path,key):
    import math
    row=read_json(path)
    values=('train_peak_reserved_mib','eval_peak_reserved_mib','rss_peak_mib',
            'train_batch_seconds','eval_batch_seconds')
    if (row.get('profile_key')!=key or not row.get('complete') or
            any(not isinstance(row.get(k),(int,float)) or not math.isfinite(row[k]) or row[k]<0 for k in values) or
            row['train_batch_seconds']<=0 or row['eval_batch_seconds']<=0):
        raise ValueError('显存预检记录无效：'+str(path))
    return row


def environment(gpu):
    e=os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',
             MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return e


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--max-parallel',type=int,default=10)
    p.add_argument('--parent-output',type=Path,default=ROOT.parent/'diagnostics/COSPEC_FULL_LR001_V1')
    p.set_defaults(windows=[7,11])
    p.add_argument('--eval-batch',type=int,choices=(128,),default=128)
    p.add_argument('--headroom-mib',type=float,default=1024.)
    p.add_argument('--check-only',action='store_true')
    return p


def prepare(args):
    from window_protocol import prepare_config
    return prepare_config(args, gpu_info(args.gpu))


def run_loop(config,args,stop):
    from results import complete,aggregate
    from window_results import package
    root=Path(config['output'])
    tasks=config['jobs']
    completed={i for i,j in enumerate(tasks) if complete(config,j)}
    workload=set(range(len(tasks)))-completed
    profiles={}
    active={}
    failed={}
    multipliers=read_json(root/'memory_corrections.json',{})
    last_snapshot=0.
    eta_model=ThroughputETA()
    max_seen=0
    adopted_count=0
    previous_failures=read_json(root/'failures.json',[])
    if previous_failures:
        write_json(root/'failure_history'/('failures_'+time.strftime('%Y%m%d_%H%M%S')+'.json'),previous_failures)
    write_json(root/'failures.json',[])

    def snapshot(state='running',eta=None,phase=None):
        nonlocal max_seen
        max_seen=max(max_seen,len(active))
        entries=[dict(index=i,label=label(tasks[i]),phase=e.get('phase','initializing'),
                      pid=e.get('progress',{}).get('pid'),adopted=e.get('adopted',False)) for i,e in active.items()]
        write_json(root/'status.json',dict(state=state,phase=phase,total=len(tasks),completed=len(completed),
                   active=len(active),max_parallel=read_json(root/'runtime_control.json',{}).get('max_parallel',args.max_parallel),
                   active_jobs=entries,eta_seconds=eta,failed=len(failed),max_observed_parallel=max_seen,
                   adopted_workers=adopted_count,
                   updated_unix=time.time(),**identity()))

    snapshot(state='preflight',phase='校验缓存和模型显存峰值')
    representatives={}
    for i,j in enumerate(tasks):
        if i in workload or args.check_only:
            representatives.setdefault(profile_key(config,j),i)
    # Profile files are hardware/backend/source keyed, and survive restarts.
    for key,index in representatives.items():
        path=root/'profiles'/(key+'.json')
        if path.exists():
            profiles[key]=load_profile(path,key)
            continue
        # Do not measure a new profile over orphaned workers from this queue.
        while any(locked(job_dir(config,j)/'run.lock') for j in tasks):
            if stop():raise SafeStop()
            snapshot(state='preflight',phase='等待上次运行的任务落盘')
            time.sleep(1)
        while locked(path.with_suffix('.lock')):
            if stop():raise SafeStop()
            time.sleep(.5)
        if path.exists():
            profiles[key]=load_profile(path,key);continue
        if stop():raise SafeStop()
        log=root/'logs'/('probe_'+key+'.log')
        for attempt in range(2):
            # The first measurement has no known peak. Measure only on an idle
            # GPU rather than turning another process's allocation into a failure.
            while True:
                if stop():raise SafeStop()
                gpu=gpu_info(args.gpu)
                if not gpu_processes(args.gpu) and gpu['free_mib']>=max(gpu['total_mib']*.9,args.headroom_mib+1024):
                    break
                snapshot(state='preflight',phase='等待空闲显卡进行显存预检')
                time.sleep(2)
            with log.open('ab',buffering=0) as sink:
                process=subprocess.Popen([sys.executable,'-u',str(ROOT/'task_worker.py'),'--config',str(root/'config.json'),
                                          '--index',str(index),'--probe',str(path)],stdout=sink,stderr=subprocess.STDOUT,
                                         env=environment(args.gpu),start_new_session=True)
            # Preflight workers also appear in the compact active-job list.
            active[index]=dict(phase='显存预检',profile_key=key,progress={})
            while process.poll() is None:
                snapshot(state='preflight',phase='显存预检 '+label(tasks[index]))
                if stop():
                    process.send_signal(signal.SIGTERM)
                    process.wait()
                    active.clear()
                    raise SafeStop()
                time.sleep(.5)
            active.clear()
            if stop():raise SafeStop()
            if process.returncode!=75:
                break
        if process.returncode or not path.exists():
            raise RuntimeError('GPU数值/显存预检失败：'+label(tasks[index])+'；查看'+str(log))
        profiles[key]=load_profile(path,key)
    for key,index in representatives.items():
        if reservation(profiles[key])+args.headroom_mib>config['gpu_identity']['total_mib']:
            raise RuntimeError('单项峰值连同安全余量超过当前GPU容量：'+label(tasks[index]))
    write_json(root/'preflight.json',dict(passed=True,profiles=len(profiles),protocol_id=config['protocol_id'],
               cache_datasets=list(config['data']),all_case_counts={ds:len(d['cases']) for ds,d in config['data'].items()}))
    if args.check_only:
        snapshot(state='checked',phase='预检通过，未开始训练')
        return
    pending=[]
    for i,j in enumerate(tasks):
        if i in completed:continue
        if locked(job_dir(config,j)/'run.lock'):
            active[i]=dict(process=None,adopted=True,profile_key=profile_key(config,j),phase='initializing',progress={})
        else:
            pending.append(i)
    retries={}
    adopted_count=sum(bool(e.get('adopted')) for e in active.values())
    recovery=read_json(root/'recovery.json',{})
    if recovery and recovery.get('protocol_id')!=config['protocol_id']:
        raise ValueError('显存恢复记录的协议不符')
    saved_exclusive=set(recovery.get('exclusive_jobs',[]))
    exclusive={i for i in workload if task_key(tasks[i]) in saved_exclusive or requires_exclusive(
        profiles[profile_key(config,tasks[i])],multipliers.get(profile_key(config,tasks[i]),1.),
        args.headroom_mib,config['gpu_identity']['total_mib'])}
    for i,e in active.items():
        e['exclusive']=i in exclusive

    def save_recovery():
        write_json(root/'recovery.json',dict(protocol_id=config['protocol_id'],
                   exclusive_jobs=sorted(task_key(tasks[i]) for i in exclusive if i not in completed)))

    save_recovery()
    runtime_cap=args.max_parallel
    last_oom=-1e12
    last_launch=-1e12
    last_recovery=0.
    while pending or active:
        now=time.monotonic()
        for index,e in list(active.items()):
            directory=job_dir(config,tasks[index])
            progress=read_json(directory/'progress.json',{})
            if progress and progress.get('recipe_id')!=recipe_id(config,tasks[index]):
                raise ValueError('运行中任务的进度身份不符')
            e['progress']=progress
            e['phase']=progress.get('phase','initializing')
            process=e.get('process')
            ended=process.poll() is not None if process is not None else not locked(directory/'run.lock')
            if not ended:continue
            del active[index]
            if complete(config,tasks[index]):
                completed.add(index)
                exclusive.discard(index)
                save_recovery()
                aggregate(config)
                continue
            outcome=read_json(directory/'outcome.json',{})
            code=process.returncode if process is not None else (75 if outcome.get('state')=='oom' else 1)
            if stop() or outcome.get('state')=='stopped' or code==130:
                if not stop():pending.append(index)
                continue
            if (code==75 or outcome.get('state')=='oom') and retries.get(index,0)<3:
                retries[index]=retries.get(index,0)+1
                key=e['profile_key']
                multipliers[key]=max(multipliers.get(key,1.)*1.3,1.3)
                if retries[index]>=2 or reservation(profiles[key],multiplier=multipliers[key])+args.headroom_mib>config['gpu_identity']['total_mib']:
                    exclusive.add(index)
                save_recovery()
                write_json(root/'memory_corrections.json',multipliers)
                runtime_cap=max(1,len(active))
                last_oom=now
                pending.append(index)
            else:
                failed[index]=dict(job=tasks[index],exit_code=code,error=outcome.get('error','工作进程异常退出'),
                                   log=str(root/'logs'/('task_'+str(index)+'.log')))
                write_json(root/'failures.json',list(failed.values()))
        if stop():
            (root/'STOP').touch(exist_ok=True)
            snapshot(state='stopping',phase='等待当前轮/测试批次保存')
            if not active:break
            time.sleep(.5)
            continue
        requested=read_json(root/'runtime_control.json',{}).get('max_parallel',args.max_parallel)
        requested=max(1,min(10,int(requested)))
        if runtime_cap<requested and now-last_oom>120 and now-last_recovery>60:
            runtime_cap+=1;last_recovery=now
        cap=1 if any(e.get('exclusive') for e in active.values()) else min(requested,runtime_cap,10)
        gpu=gpu_info(args.gpu)
        apps=gpu_processes(args.gpu)
        active_memory=[]
        for e in active.values():
            p=e['progress']
            # Unknown PID mapping is conservatively treated as unpaid capacity.
            current=apps.get(p.get('proc_pid'),0.)
            active_memory.append(dict(profile_key=e['profile_key'],phase=e['phase'],actual_gpu_mib=current,
                                      multiplier=1. if e.get('exclusive') else multipliers.get(e['profile_key'],1.)))
        if len(active)<cap and now-last_launch>=RAMP_SECONDS:
            for position,index in enumerate(pending):
                key=profile_key(config,tasks[index])
                if index in exclusive and active:
                    continue
                if not can_admit(profiles[key],active_memory,profiles,gpu,args.headroom_mib,
                                 1. if index in exclusive else multipliers.get(key,1.),available_ram()):
                    continue
                pending.pop(position)
                directory=job_dir(config,tasks[index]);directory.mkdir(parents=True,exist_ok=True)
                with (root/'logs'/('task_'+str(index)+'.log')).open('ab',buffering=0) as sink:
                    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'task_worker.py'),'--config',str(root/'config.json'),
                                            '--index',str(index)],stdout=sink,stderr=subprocess.STDOUT,
                                           env=environment(args.gpu),start_new_session=True)
                active[index]=dict(process=child,profile_key=key,phase='initializing',progress={},adopted=False,
                                   exclusive=index in exclusive)
                max_seen=max(max_seen,len(active))
                if len(active)>10:
                    raise RuntimeError('并行硬上限保护触发')
                last_launch=now
                break
        if now-last_snapshot>=10:
            done_units=remaining=0.
            for i,j in enumerate(tasks):
                if i in failed or i not in workload:continue
                cost=estimated_work(config,j,profiles[profile_key(config,j)])
                p=active[i]['progress'] if i in active else read_json(job_dir(config,j)/'progress.json',{})
                done=work_done(cost,p,i in completed)
                done_units+=done
                remaining+=cost['total']-done
            eta=eta_model.update(now,done_units,remaining)
            snapshot(eta=eta,phase='等待显存' if not active and pending else '训练与16条件测试')
            last_snapshot=now
        time.sleep(POLL_SECONDS)
    state='stopped' if stop() else ('completed' if len(completed)==len(tasks) and not failed else 'partial')
    snapshot(state='finalizing',eta=None,phase='汇总与打包')
    aggregate(config)
    from window_results import combined_table
    combined_table(config)
    for tag in ('results','maps','features'):
        package(config,tag)
    snapshot(state=state,eta=0 if state=='completed' else None,phase='全部完成' if state=='completed' else state)


def main():
    args=parser().parse_args()
    from window_protocol import protect_output
    protect_output(args.output,args.reference,args.parent_output)
    root=args.output.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('已有调度器运行，没有重复启动')
        flag=[False]
        signal.signal(signal.SIGTERM,lambda *_:flag.__setitem__(0,True))
        signal.signal(signal.SIGINT,lambda *_:flag.__setitem__(0,True))
        stop=lambda:flag[0] or (root/'STOP').exists()
        try:
            config=prepare(args)
            run_loop(config,args,stop)
            return 0
        except SafeStop:
            status=read_json(root/'status.json',{})
            status.update(state='stopped',active=0,active_jobs=[],phase='预检已停止',updated_unix=time.time())
            write_json(root/'status.json',status)
            return 0
        except Exception as error:
            traceback.print_exc()
            (root/'STOP').write_text('controller failed; save own workers safely\n')
            status=read_json(root/'status.json',{})
            status.update(state='failed',error=str(error),updated_unix=time.time())
            write_json(root/'status.json',status)
            return 1


if __name__=='__main__':
    raise SystemExit(main())
