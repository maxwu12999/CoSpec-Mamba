#!/usr/bin/env python3
"""No real training data required. CPU state/resume tests and optional GPU numerics."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import contextlib
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch import nn
from common import (ROOT,read_json,write_json,make_jobs,task_key,job_dir,profile_key,recipe_id,SafeStop,verify_frozen,file_hash)
from data import all_cases,mask_hash
from models import configure_imports,build_model
configure_imports()
from external_common import seed_all,state_hash
from external_metrics import classification_metrics
from worker import Metrics,Progress,train,evaluate,run
from policy import can_admit,ThroughputETA


class TinyNet(nn.Module):
    def __init__(self,channels,classes):
        super().__init__()
        self.features=nn.Sequential(nn.Linear(channels,12),nn.BatchNorm1d(12),nn.SiLU(),nn.Dropout(.2))
        self.head=nn.Linear(12,classes)

    def forward(self,x,observed):
        measured=torch.where(observed[:,:,None,None],x,torch.zeros_like(x)).mean((-1,-2))
        embedding=self.features(measured)
        return dict(logits=self.head(embedding),embedding=embedding)


def tiny_factory(config,job):
    d=config['data'][job['dataset']]
    return TinyNet(d['shape'][-1],d['classes'])


def fixture(root):
    cache=Path(root)/'reference/cache/PaviaU';cache.mkdir(parents=True)
    rng=np.random.default_rng(42)
    cube=rng.normal(size=(16,16,32)).astype(np.float32)
    tr=np.zeros((16,16),np.int64);te=np.zeros_like(tr)
    tr.ravel()[:24]=np.tile(np.arange(1,4),8)
    te.ravel()[50:80]=np.tile(np.arange(1,4),10)
    wave=np.linspace(0,1,32,dtype=np.float32)
    for n,a in [('cube',cube),('tr',tr),('te',te),('wave',wave)]:np.save(cache/(n+'.npy'),a)
    np.savez(cache/'stats.npz',means=np.stack([cube[tr==c].mean(0) for c in (1,2,3)]),variances=np.ones((3,32),np.float32))
    cases=[dict(name='full',fraction=1.,repeat=-1,mask=[True]*32)]
    ordering=np.random.default_rng(20261014)
    for repeat in (1,2,3):
        order=ordering.permutation(32)
        for f in (.8,.6,.4):
            mask=np.zeros(32,bool);mask[order[:round(32*f)]]=True
            cases.append(dict(name=f'keep{round(f*100)}_r{repeat}',fraction=f,repeat=repeat,mask=mask.tolist()))
    for c in cases:c['sha256']=mask_hash(c['mask'])
    meta=dict(dataset='PaviaU',shape=[16,16,32],classes=3,train_count=24,test_count=30,
              class_support=[10,10,10],cache=str(cache),cases=cases,data_id='synthetic_only',
              normalization=dict(fit_on='synthetic_training_centers_only'))
    meta['cases']=all_cases(meta)
    meta['cache_sha256']={n:file_hash(cache/n) for n in ('cube.npy','tr.npy','te.npy','wave.npy','stats.npz')}
    write_json(cache/'data.json',meta)
    pca=cache/'pca.npz'
    np.savez(pca,mean=np.zeros(32,np.float32),projection=np.eye(32,dtype=np.float32)[:30])
    return dict(output=str(Path(root)/'output'),protocol_id='pipeline_check_only',scope='pipeline_check',
                source_sha256='synthetic_only',data={'PaviaU':meta},device='cpu',
                gpu_identity={'name':'CPU fixture','total_mib':24564.},pca={'PaviaU':{'path':str(pca)}},
                backend={'suite':'reference','external':'native'},
                train=dict(epochs=100,batch=7,eval_batch=8,lr=.001,lr_policy='constant'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=p.parse_args();torch.set_num_threads(1)
    verify_frozen()
    jobs=make_jobs();assert len(jobs)==210 and len(set(map(task_key,jobs)))==210
    assert sum(j['group']=='suite99' for j in jobs)==99
    assert sum(j['group']=='external105' for j in jobs)==105
    assert sum(j['group']=='window' and j['patch']==9 for j in jobs)==6
    assert len(make_jobs((7,9,11)))==222
    for ds,meta in read_json(ROOT/'REFERENCE_DATA.json').items():
        cases=all_cases(meta);assert len(cases)==16
        for repeat in (1,2,3):
            masks=[np.array(next(c['mask'] for c in cases if c['name']==f'keep{k}_r{repeat}')) for k in (10,20,40,60,80)]
            assert all(np.all(a<=b) for a,b in zip(masks,masks[1:]))
    print('[通过] 99+105+6=210项；7个数据集的16套掩码正确共享和嵌套')
    rng=np.random.default_rng(123);truth=rng.integers(0,5,137);prob=rng.random((137,5));prob/=prob.sum(1,keepdims=True)
    acc=Metrics(5)
    for start in range(0,137,13):acc.add(truth[start:start+13],prob[start:start+13])
    ours=acc.finish();reference=classification_metrics(truth,prob,5)
    for key,value in reference.items():np.testing.assert_allclose(ours[key],value,rtol=1e-12,atol=1e-12)
    print('[通过] 流式指标与原版OA/AA/Kappa/NLL/Brier/ECE/混淆矩阵一致')
    profile=dict(train_peak_reserved_mib=4000.,eval_peak_reserved_mib=7000.,rss_peak_mib=500.)
    profiles={'p':profile};gpu=dict(free_mib=24000.,total_mib=24564.)
    assert can_admit(profile,[],profiles,gpu)
    assert not can_admit(profile,[dict(profile_key='p',actual_gpu_mib=0.),dict(profile_key='p',actual_gpu_mib=0.)],profiles,gpu)
    assert not can_admit(profile,[],profiles,dict(free_mib=7000.,total_mib=24564.))
    assert not can_admit(profile,[],profiles,gpu,available_ram=500.)
    eta=ThroughputETA();assert eta.update(0.,0.,100.) is None
    assert abs(eta.update(10.,5.,95.)-190.)<1e-8
    print('[通过] 显存峰值及启动预约、RAM余量、按总吞吐估时逻辑')
    with tempfile.TemporaryDirectory(prefix='cospec_full_check_') as temporary:
        root=Path(temporary);cfg=fixture(root)
        from data import validate_caches,prepare_pca
        validated=validate_caches(root/'reference',root/'separate_output',datasets=('PaviaU',),synthetic=True)
        prepared=prepare_pca(validated,root/'separate_output')
        assert len(validated['PaviaU']['cases'])==16 and 'PaviaU' in prepared
        print('[通过] 原缓存校验、中心互斥、PCA训练中心拟合及重启复用路径')
        job=dict(dataset='PaviaU',model='D',seed=2,patch=15,group='pipeline_check')
        cfg['jobs']=[job]
        def fresh(folder):
            directory=root/folder;directory.mkdir()
            seed_all(job['seed']);model=tiny_factory(cfg,job)
            progress=Progress(directory/'progress.json',recipe_id(cfg,job))
            return directory,model,progress
        with (root/'test.log').open('w') as log,contextlib.redirect_stdout(log):
            ad,am,ap=fresh('continuous');ah=train(am,cfg,job,ad,ap,lambda:False)
            bd,bm,bp=fresh('resumed')
            try:train(bm,cfg,job,bd,bp,lambda:False,stop_after_epoch=73)
            except SafeStop:pass
            else:raise AssertionError('没有在73轮停止')
            seed_all(job['seed']);bm=tiny_factory(cfg,job)
            bh=train(bm,cfg,job,bd,bp,lambda:False)
            assert state_hash(am.state_dict())==state_hash(bm.state_dict())
            assert all(r['lr']==.001 for r in bh.values()) and len(bh)==100
            sha=state_hash(am.state_dict())
            full=evaluate(am,cfg,job,ad,sha,ap,lambda:False)
            try:evaluate(bm,cfg,job,bd,sha,bp,lambda:True)
            except SafeStop:pass
            else:raise AssertionError('测试没有安全停止')
            cursor=read_json(bd/'partial/full/cursor.json');assert cursor['next_pixel']==8
            resumed=evaluate(bm,cfg,job,bd,sha,bp,lambda:False)
            for a,b in zip(full,resumed):
                for key in ('oa_percent','aa_percent','kappa','confusion_matrix','count','nll','brier'):
                    np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)
        print('[通过] 固定0.001完整100轮，第73轮恢复与连续训练的CPU权重完全相同')
        print('[通过] 测试在第一个批次中断并续跑，16个条件的结果与连续测试一致')
        if args.device=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
            cfg['device']='cuda';cfg['backend']=dict(suite='cuda',external='cuda_fused')
        names=['A','B','C','D','D_structure','D_row','D_raster4','D_random4','ConvMamba_aug',
               'HybridSN','A2S2K','SQSFormer','GSCViT','MiM_v2']
        from data import Patches
        x,y=Patches(cfg['data']['PaviaU'],'tr',15).batch(range(2))
        x,y=x.to(cfg['device']),y.to(cfg['device'])
        from torch.nn import functional as F
        for name in names:
            j=dict(job,model=name);seed_all(2);m=build_model(cfg,j).to(cfg['device'])
            mask=torch.ones((2,32),dtype=torch.bool,device=cfg['device']);mask[:,3::4]=False
            out=m(x,mask)['logits'];assert out.shape==(2,3) and torch.isfinite(out).all()
            F.cross_entropy(out,y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),15.,error_if_nonfinite=True)
            del m,out
            print('[通过] 实际模型15×15前后向：'+name,flush=True)
        for name in ('B','D'):
            j=dict(job,model=name,patch=9);m=build_model(cfg,j).to(cfg['device'])
            xx,yy=Patches(cfg['data']['PaviaU'],'tr',9).batch(range(2))
            out=m(xx.to(cfg['device']),mask)['logits'];assert out.shape==(2,3)
            F.cross_entropy(out,yy.to(cfg['device'])).backward();del m,out
        print('[通过] 9×9的B/D实际模型前后向')
    print('全部自检完成；CPU逐位恢复测试不被用作CUDA逐位一致性要求。')


if __name__=='__main__':main()
