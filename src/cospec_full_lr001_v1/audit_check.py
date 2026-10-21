#!/usr/bin/env python3
"""Regression checks for the reviewed scheduler repair; no GPU or user data."""
import argparse
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
from common import (ROOT,read_json,write_json,digest,make_jobs,job_dir,profile_key,recipe_id,SafeStop)
from protocol_compat import scientific_guard,resolve_protocol
from policy import reservation,requires_exclusive,can_admit
import scheduler


def minimal(root):
    cases=[]
    for i in range(16):
        fraction=1. if i==0 else (.8,.6,.4,.2,.1)[(i-1)%5]
        repeat=-1 if i==0 else 1+(i-1)//5
        cases.append(dict(name='full' if i==0 else f'keep{round(fraction*100)}_r{repeat}',
                          fraction=fraction,repeat=repeat,sha256=str(i)))
    job=dict(dataset='PaviaU',model='D',seed=1,patch=15,group='audit_only')
    config=dict(output=str(root),protocol_id='audit_only',source_sha256='synthetic_only',scope='audit_only',
                train=dict(epochs=100,batch=32,eval_batch=128,lr=.001,lr_policy='constant'),
                data={'PaviaU':dict(shape=[10,10,32],classes=3,train_count=24,test_count=30,
                                    class_support=[10,10,10],cases=cases)},
                jobs=[job],backend={'suite':'reference','external':'native'},
                gpu_identity={'name':'simulation','total_mib':24564.})
    root.mkdir(exist_ok=True,parents=True);(root/'logs').mkdir();(root/'profiles').mkdir()
    write_json(root/'config.json',config)
    write_json(root/'runtime_control.json',dict(max_parallel=10))
    return config,job,scheduler.parser().parse_args(['--output',str(root)])


def valid_result(config,job):
    metrics=[dict(case=c['name'],fraction=c['fraction'],repeat=c['repeat'],mask_sha256=c['sha256'],
                  count=30,class_support=[10,10,10],oa_percent=70.,aa_percent=70.,kappa=.55)
             for c in config['data']['PaviaU']['cases']]
    return dict(recipe_id=recipe_id(config,job),complete=True,job=job,lr_policy='constant_0.001',
                completed_epochs=100,metrics=metrics)


def main():
    p=argparse.ArgumentParser();p.add_argument('--torch',action='store_true');args=p.parse_args()
    scientific_guard()
    protocol=dict(version='cospec_full_lr001_v1',source_sha256='current_audit_source',scope='formal',
                  jobs=make_jobs(),train=dict(lr=.001,epochs=100),backend={'suite':'cuda'},
                  data_identity={'PaviaU':'fixed_data_and_masks'},pca_identity={'PaviaU':'fixed_pca'})
    old=copy.deepcopy(protocol);old['source_sha256']=read_json(ROOT/'COMPATIBILITY.json')['reviewed_legacy_source_sha256'][0]
    old['protocol_id']=digest(old)
    result,audit=resolve_protocol(protocol,old)
    assert audit['used'] and digest(result)==old['protocol_id']
    for key,value in [('train',dict(lr=.0001,epochs=100)),('jobs',[]),('data_identity',{'PaviaU':'changed'})]:
        changed=copy.deepcopy(protocol);changed[key]=value
        result,audit=resolve_protocol(changed,old)
        assert not audit['used'] and digest(result)!=old['protocol_id']
    with patch('protocol_compat.file_hash',return_value='changed'):
        try:resolve_protocol(protocol,old)
        except ValueError:pass
        else:raise AssertionError('changed scientific files were accepted')
    print('[通过] 更新保留首版固定0.001的协议和断点；改变学习率、清单、数据或科学代码时拒绝混用')
    profile=dict(train_peak_reserved_mib=4000.,eval_peak_reserved_mib=8000.,rss_peak_mib=300.,
                 train_batch_seconds=.01,eval_batch_seconds=.002)
    assert requires_exclusive(profile,3.,1024.,24564.)
    assert not can_admit(profile,[],{},dict(free_mib=24564.),multiplier=3.)
    assert can_admit(profile,[],{},dict(free_mib=24564.),multiplier=1.)
    original_read=Path.read_text
    def fake_read(path,*a,**kw):
        known={'/proc/meminfo':'MemAvailable: 67108864 kB\n',
               '/sys/fs/cgroup/memory.max':str(4*2**30),'/sys/fs/cgroup/memory.current':str(3*2**30)}
        if str(path) in known:return known[str(path)]
        raise FileNotFoundError(str(path))
    with patch.object(Path,'read_text',fake_read):assert scheduler.available_ram()==1024.
    print('[通过] 超大OOM修正可降为独占重试；RAM保护采用容器额度')
    with tempfile.TemporaryDirectory(prefix='cospec_audit_regression_') as tmp:
        root=Path(tmp)
        from data import all_cases
        ds=read_json(ROOT/'REFERENCE_DATA.json')
        for name,meta in ds.items():
            meta['cases']=all_cases(meta);meta['cache']=str(root/'unused_original_cache'/name)
        prep={name:dict(path=str(root/'unused_pca'/name/'pca.npz'),sha256='mock_pca_'+name) for name in ds}
        ga=scheduler.parser().parse_args(['--output',str(root/'upgrade_prepare')])
        hardware=dict(name='simulation',uuid='fake',total_mib=24564.,free_mib=24564.,driver='test')
        with patch('data.validate_caches',return_value=ds),patch('data.prepare_pca',return_value=prep), \
             patch.object(scheduler,'gpu_info',return_value=hardware), \
             patch.object(scheduler.importlib.metadata,'version',return_value='2.5.1'):
            fresh=scheduler.prepare(ga)
            original=copy.deepcopy(fresh)
            original['source_sha256']=old['source_sha256']
            fields=('version','source_sha256','scope','jobs','train','backend','data_identity','pca_identity')
            original['protocol_id']=digest({k:original[k] for k in fields})
            original.pop('runtime_source_sha256',None);original.pop('protocol_compatibility',None)
            write_json(ga.output/'config.json',original)
            checkpoint=job_dir(original,original['jobs'][0])/'last.pt'
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'checkpoint-must-stay-unchanged')
            upgraded=scheduler.prepare(ga)
            assert upgraded['protocol_id']==original['protocol_id'] and upgraded['protocol_compatibility']['used']
            assert checkpoint.read_bytes()==b'checkpoint-must-stay-unchanged'
            assert list(ga.output.glob('configuration_backup_*.json'))
        print('[通过] 实际prepare更新路径保留首版协议并备份配置，未重写已有检查点')
        cfg,job,a=minimal(root/'case_guard')
        directory=job_dir(cfg,job);directory.mkdir(parents=True)
        path=directory/'result.json';good=valid_result(cfg,job);write_json(path,good)
        from results import complete
        assert complete(cfg,job)
        for broken in ('duplicate','repeat','support'):
            bad=copy.deepcopy(good)
            if broken=='duplicate':bad['metrics'][-1]=copy.deepcopy(bad['metrics'][0])
            if broken=='repeat':bad['metrics'][1]['repeat']=99
            if broken=='support':bad['metrics'][1]['class_support']=[9,10,11]
            write_json(path,bad)
            try:complete(cfg,job)
            except ValueError:pass
            else:raise AssertionError('invalid completed result was accepted: '+broken)
        write_json(path,good)
        print('[通过] 完成验收拒绝重复条件、错误重复号和错误类别数量')
        write_json(root/'case_guard/status.json',dict(state='running',active_jobs=[{'label':'stale job'}],
                  total=1,completed=0,updated_unix=0))
        from dashboard import frame,watch
        text,state,alive=frame(root/'case_guard')
        assert state=='interrupted' and not alive and 'stale job' not in text and len(text.splitlines())==4
        with contextlib.redirect_stdout(io.StringIO()):watch(root/'case_guard')
        print('[通过] 调度器意外退出后，界面不再将旧快照显示为仍在运行')
        cfg2,job2,a2=minimal(root/'busy_probe')
        checks=[0]
        def stop():checks[0]+=1;return checks[0]>=4
        with patch.object(scheduler,'gpu_info',return_value=dict(free_mib=1000.,total_mib=24564.)), \
             patch.object(scheduler,'gpu_processes',return_value={123:20000.}), \
             patch.object(scheduler.subprocess,'Popen') as launch,patch.object(scheduler.time,'sleep'):
            try:scheduler.run_loop(cfg2,a2,stop)
            except SafeStop:pass
            else:raise AssertionError('busy GPU wait was not stoppable')
            launch.assert_not_called()
        assert read_json(Path(cfg2['output'])/'status.json')['phase']=='等待空闲显卡进行显存预检'
        print('[通过] 其他进程占卡时等待，等待期间可停止，不抢跑显存预检')
        cfg3,job3,a3=minimal(root/'cached_impossible')
        key=profile_key(cfg3,job3)
        write_json(Path(cfg3['output'])/'profiles'/(key+'.json'),dict(profile,complete=True,profile_key=key,
                   eval_peak_reserved_mib=30000.))
        try:scheduler.run_loop(cfg3,a3,lambda:False)
        except RuntimeError as e:assert '超过' in str(e)
        else:raise AssertionError('impossible cached peak waited forever')
        print('[通过] 已缓存的单项峰值超出卡容量时明确报错，不永久等待')
        write_json(Path(cfg['output'])/'failures.json',[{'previous_failure':'recovered'}])
        with patch.object(scheduler,'gpu_info',side_effect=AssertionError('completed runs must not re-profile')):
            scheduler.run_loop(cfg,a,lambda:False)
        assert read_json(Path(cfg['output'])/'status.json')['state']=='completed'
        assert read_json(Path(cfg['output'])/'failures.json')==[]
        assert list((Path(cfg['output'])/'failure_history').glob('*.json'))
        print('[通过] 全部完成后再启动只汇总打包；旧失败记录留档，不误报当前失败')
        if args.torch:
            import torch
            torch.set_num_threads(1)
            from self_check import fixture,tiny_factory
            import worker
            cfg4=fixture(root/'probe_stop');j=dict(dataset='PaviaU',model='D',seed=1,patch=15,group='audit_only')
            target=root/'probe_stop/stopped_profile.json'
            stopped=[0]
            def stop_probe():stopped[0]+=1;return stopped[0]>=3
            with patch.object(worker,'build_model',side_effect=tiny_factory):
                try:worker.memory_probe(cfg4,j,target,stop_probe)
                except SafeStop:pass
                else:raise AssertionError('probe ignored stop callback')
            assert not target.exists()
            print('[通过] 实际小网络的显存预检响应停止，不写入未完成的峰值记录')
    print('审计回归检查全部通过。')


if __name__=='__main__':main()
