#!/usr/bin/env python3
"""Temporary CPU models and simulated GPU; original scientific files stay intact."""
from bootstrap import use_core,ADDON
use_core()
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
from common import read_json,write_json,job_dir,profile_key,file_hash,locked,source_hash
from self_check import fixture,tiny_factory
import worker
import scheduler
from speed_runtime import metadata,record_source


def main():
    import torch
    torch.set_num_threads(1)
    initial=source_hash()
    with tempfile.TemporaryDirectory(prefix='cospec_speed_process_') as tmp:
        root=Path(tmp);cfg=fixture(root/'fixture');cfg['train']['epochs']=2
        cfg['jobs']=[dict(dataset='PaviaU',model='D',seed=i+1,patch=15,group='speed_process_check') for i in range(12)]
        cfg['runtime_speedup']=metadata()
        output=Path(cfg['output']);output.mkdir();(output/'logs').mkdir();(output/'profiles').mkdir()
        write_json(output/'config.json',cfg);write_json(output/'runtime_control.json',dict(max_parallel=10))
        record_source(cfg)
        worker.run(cfg,cfg['jobs'][0],lambda:False,factory=tiny_factory)
        old_result=job_dir(cfg,cfg['jobs'][0])/'result.json';old_hash=file_hash(old_result)
        key=profile_key(cfg,cfg['jobs'][0])
        write_json(output/'profiles'/(key+'.json'),dict(complete=True,profile_key=key,train_peak_reserved_mib=64.,
                   eval_peak_reserved_mib=64.,rss_peak_mib=300.,train_batch_seconds=.01,eval_batch_seconds=.002))
        wrapper=root/'wrapper';wrapper.mkdir()
        (wrapper/'worker.py').write_text(f'''import sys
sys.path.insert(0,{str(ADDON)!r})
from bootstrap import use_core
use_core()
import worker
from self_check import tiny_factory
from speed_runtime import install
install(prefetch=True)
original=worker.run
def run(*args,**kwargs):
    kwargs['factory']=tiny_factory
    return original(*args,**kwargs)
worker.run=run
raise SystemExit(worker.main())
''')
        with patch.object(scheduler,'ROOT',wrapper),patch.object(scheduler,'RAMP_SECONDS',0.), \
             patch.object(scheduler,'POLL_SECONDS',.02), \
             patch.object(scheduler,'gpu_info',return_value=dict(total_mib=24564.,free_mib=24564.)), \
             patch.object(scheduler,'gpu_processes',return_value={}), \
             patch.object(scheduler,'available_ram',return_value=102400.):
            args=scheduler.parser().parse_args(['--output',str(output)])
            deadline=time.monotonic()+170
            scheduler.run_loop(cfg,args,lambda:time.monotonic()>deadline)
        status=read_json(output/'status.json')
        if status['completed']!=12:
            for path in (output/'logs').glob('*.log'):print(path.name,path.read_text()[-1800:])
            raise AssertionError(status)
        assert status['max_observed_parallel']==10,status
        assert file_hash(old_result)==old_hash
        for job in cfg['jobs'][1:]:
            result=read_json(job_dir(cfg,job)/'result.json')
            assert result['runtime']['speedup']['kind']=='cospec_speedup_v1'
        assert (output.with_name(output.name+'_results.zip')).exists()
        print('[通过] 新入口模拟10路进程完成；已完成项不重训、不改写，结果和运行扩展源码均留档',flush=True)
        spec=importlib.util.spec_from_file_location('speed_launcher_test',ADDON/'launcher.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        saved_argv=sys.argv[:];saved_root=launcher.control.ROOT
        try:
            # An orphan worker holds only a task lock; switch must still wait.
            path=job_dir(cfg,cfg['jobs'][0])/'run.lock'
            code="import fcntl,time;from pathlib import Path;f=open("+repr(str(path))+",'a+');fcntl.flock(f,fcntl.LOCK_EX);time.sleep(.5)"
            orphan=subprocess.Popen([sys.executable,'-c',code])
            until=time.monotonic()+5
            while not locked(path) and time.monotonic()<until:time.sleep(.01)
            assert locked(path)
            def launch():
                assert not locked(path)
                assert not (output/'STOP').exists()
                assert launcher.control.ROOT==ADDON
                assert '--max-parallel' in sys.argv
                return 0
            sys.argv=[str(ADDON/'launcher.py'),'--headless','--output',str(output),'--max-parallel','5']
            with patch.object(launcher.control,'main',side_effect=launch):assert launcher.main()==0
            assert orphan.wait()==0
            def rollback():
                assert launcher.control.ROOT==launcher.CORE
                assert '--original' not in sys.argv
                assert not (output/'STOP').exists()
                return 0
            sys.argv=[str(ADDON/'launcher.py'),'--original','--headless','--output',str(output)]
            with patch.object(launcher.control,'main',side_effect=rollback):assert launcher.main()==0
        finally:
            sys.argv=saved_argv;launcher.control.ROOT=saved_root
        print('[通过] 切换等待真实工作锁释放后才启动新队列，包含无父调度器的工作进程',flush=True)
        spec=importlib.util.spec_from_file_location('speed_scheduler_test',ADDON/'scheduler.py')
        bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
        try:
            for enabled in (True,False):
                c=fixture(root/('gate_'+str(enabled)));o=Path(c['output']);o.mkdir()
                c['jobs']=[dict(cfg['jobs'][0],group='gate_check')]
                c['data_identity']={'PaviaU':{'test':'synthetic_only'}}
                write_json(o/'config.json',c)
                a=scheduler.parser().parse_args(['--output',str(o)])
                def probe(command,**kwargs):
                    target=Path(command[command.index('--report')+1])
                    write_json(target,dict(passed=True,enabled=enabled,weighted_test_speedup=2. if enabled else .9))
                    return subprocess.CompletedProcess(command,0)
                original_root=bridge.core_scheduler.ROOT;bridge.core_scheduler.ROOT=ADDON
                with patch.object(bridge,'original_prepare',return_value=c),patch.object(bridge.subprocess,'run',side_effect=probe):
                    after=bridge.prepare(a)
                assert after['protocol_id']==c['protocol_id']
                assert after['runtime_speedup']['enabled']==enabled
                assert bridge.core_scheduler.ROOT==(ADDON if enabled else launcher.CORE)
                bridge.core_scheduler.ROOT=original_root
        finally:pass
        print('[通过] GPU对照/测速门控：确认收益才启用；无收益回到原工作入口，协议ID不变',flush=True)
    assert source_hash()==initial
    print('进程级加速入口检查全部完成。')


if __name__=='__main__':main()
