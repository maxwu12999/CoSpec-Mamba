"""Internal process-level scheduler test. Synthetic CPU jobs, simulated GPU info."""
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import time
PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE))
from common import read_json,write_json,profile_key,job_dir,locked
from self_check import fixture
import scheduler

with tempfile.TemporaryDirectory(prefix='cospec_full_scheduler_test_') as temporary:
    root=Path(temporary)
    config=fixture(root)
    config['train']['epochs']=2
    config['jobs']=[dict(dataset='PaviaU',model='D',seed=i+1,patch=15,group='pipeline_check') for i in range(12)]
    output=Path(config['output']);output.mkdir()
    (output/'logs').mkdir();(output/'profiles').mkdir()
    write_json(output/'config.json',config)
    write_json(output/'runtime_control.json',dict(max_parallel=10))
    key=profile_key(config,config['jobs'][0])
    write_json(output/'profiles'/(key+'.json'),dict(complete=True,profile_key=key,
               train_peak_reserved_mib=64.,eval_peak_reserved_mib=64.,train_batch_seconds=.01,
               eval_batch_seconds=.002,rss_peak_mib=300.))
    wrapper=root/'wrapper';wrapper.mkdir()
    (wrapper/'worker.py').write_text(f'''import sys,time,os
sys.path.insert(0,{str(PACKAGE)!r})
from pathlib import Path
from common import read_json,write_json,job_dir,recipe_id
import worker
from self_check import tiny_factory
args=worker.argparse.ArgumentParser();args.add_argument('--config',type=Path);args.add_argument('--index',type=int);a=args.parse_args()
config=read_json(a.config);job=config['jobs'][a.index]
directory=job_dir(config,job);directory.mkdir(parents=True,exist_ok=True)
marker=directory/'oom_injected'
if a.index==0 and not marker.exists():
    marker.write_text('intentional simulated OOM, not a GPU failure')
    time.sleep(3)
    write_json(directory/'outcome.json',dict(state='oom',error='simulated out of memory',recipe_id=recipe_id(config,job)))
    raise SystemExit(75)
old=worker.run
def factory(config,job):
    time.sleep(float(os.environ.get('COSPEC_TEST_HOLD_INIT','0')))
    return tiny_factory(config,job)
def run(*args,**kwargs):
    kwargs['factory']=factory
    return old(*args,**kwargs)
worker.run=run
raise SystemExit(worker.main())
''',encoding='utf-8')
    scheduler.ROOT=wrapper
    scheduler.RAMP_SECONDS=0.
    scheduler.POLL_SECONDS=.02
    scheduler.gpu_info=lambda index:dict(name='simulated',uuid='sim',total_mib=24564.,free_mib=24564.,driver='test')
    scheduler.gpu_processes=lambda index:{}
    scheduler.available_ram=lambda:102400.
    args=scheduler.parser().parse_args(['--output',str(output)])
    deadline=time.monotonic()+170
    try:
        scheduler.run_loop(config,args,lambda:time.monotonic()>deadline)
        status=read_json(output/'status.json')
        if status['state']!='completed':
            for p in (output/'logs').glob('*.log'):print(p.name,p.read_text()[-1800:])
            raise AssertionError(status)
        assert status['completed']==status['total']==12
        assert status['max_observed_parallel']==10,status
        assert read_json(output/'memory_corrections.json')[key]>1
        assert len(list(output.glob('runs/**/result.json')))==12
        assert (output.with_name(output.name+'_results.zip')).exists()
        assert (output.with_name(output.name+'_maps.zip')).exists()
        assert (output.with_name(output.name+'_features.zip')).exists()
        from dashboard import frame
        text,_,_=frame(output);assert len(text.splitlines())==4
        print('[集成通过] 最大10路、12个进程任务完成、OOM降并行重试、16条件和分类图/特征/结果ZIP')
        print('[集成通过] 主界面仅4项；完成项不重复计数')
        def ready_case(path,count):
            cfg=fixture(path);cfg['train']['epochs']=2
            cfg['jobs']=[dict(dataset='PaviaU',model='D',seed=i+1,patch=15,group='pipeline_check') for i in range(count)]
            out=Path(cfg['output']);out.mkdir();(out/'logs').mkdir();(out/'profiles').mkdir()
            write_json(out/'config.json',cfg);write_json(out/'runtime_control.json',dict(max_parallel=10))
            k=profile_key(cfg,cfg['jobs'][0])
            write_json(out/'profiles'/(k+'.json'),dict(complete=True,profile_key=k,train_peak_reserved_mib=64.,
                       eval_peak_reserved_mib=64.,train_batch_seconds=.01,eval_batch_seconds=.002,rss_peak_mib=300.))
            return cfg,out,scheduler.parser().parse_args(['--output',str(out)])
        cfg2,out2,args2=ready_case(root/'stop_case',3)
        stopping_time=time.monotonic()+.1
        scheduler.run_loop(cfg2,args2,lambda:time.monotonic()>stopping_time)
        assert read_json(out2/'status.json')['state']=='stopped'
        assert not any(locked(job_dir(cfg2,j)/'run.lock') for j in cfg2['jobs'])
        (out2/'STOP').unlink()
        scheduler.run_loop(cfg2,args2,lambda:False)
        assert read_json(out2/'status.json')['completed']==3
        print('[集成通过] 调度器停止会通知本队列工作进程保存，随后原配置可续跑')
        cfg3,out3,args3=ready_case(root/'adopt_case',1)
        directory=job_dir(cfg3,cfg3['jobs'][0]);directory.mkdir(parents=True)
        (directory/'oom_injected').touch()
        env=os.environ.copy();env['COSPEC_TEST_HOLD_INIT']='2'
        with (out3/'adopted.log').open('w') as log:
            orphan=subprocess.Popen([sys.executable,str(wrapper/'worker.py'),'--config',str(out3/'config.json'),'--index','0'],
                                    stdout=log,stderr=subprocess.STDOUT,env=env)
            until=time.monotonic()+20
            while not locked(directory/'run.lock') and time.monotonic()<until:time.sleep(.02)
            assert locked(directory/'run.lock')
            scheduler.run_loop(cfg3,args3,lambda:False)
            assert orphan.wait()==0
        state3=read_json(out3/'status.json');assert state3['adopted_workers']==1 and state3['completed']==1
        print('[集成通过] 接管仍在运行的工作进程，不重复启动同一实验')
        cfg4,out4,args4=ready_case(root/'exclusive_restart_case',2)
        k4=profile_key(cfg4,cfg4['jobs'][0])
        write_json(out4/'profiles'/(k4+'.json'),dict(complete=True,profile_key=k4,train_peak_reserved_mib=4000.,
                   eval_peak_reserved_mib=8000.,train_batch_seconds=.01,eval_batch_seconds=.002,rss_peak_mib=300.))
        write_json(out4/'memory_corrections.json',{k4:3.})
        for j in cfg4['jobs']:
            d=job_dir(cfg4,j);d.mkdir(parents=True);(d/'oom_injected').touch()
        scheduler.run_loop(cfg4,args4,lambda:False)
        s4=read_json(out4/'status.json')
        assert s4['completed']==2 and s4['max_observed_parallel']==1,s4
        assert read_json(out4/'recovery.json')['exclusive_jobs']==[]
        print('[集成通过] 带超大OOM修正的重启会自动独占完成，未陷入永久等待显存')

    except Exception:
        (output/'STOP').touch()
        for p in (output/'logs').glob('*.log'):print(p.name,p.read_text()[-2000:])
        raise
