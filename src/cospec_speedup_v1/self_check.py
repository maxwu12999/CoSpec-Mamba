#!/usr/bin/env python3
"""Exact CPU equivalence and bidirectional checkpoint/cursor compatibility."""
from bootstrap import use_core
use_core()
import argparse
import contextlib
import copy
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
import torch
from common import read_json,write_json,recipe_id,digest,SafeStop,source_hash
from data import Patches
from self_check import fixture,tiny_factory
import worker
from external_common import seed_all,state_hash
from models import build_model
import speed_data,speed_runtime,speed_eval


def compare_metrics(left,right):
    ignore={'seconds_this_segment','timing_scope'}
    for a,b in zip(left,right):
        assert a['case']==b['case']
        for key in set(a)-ignore:
            if a[key] is None:assert b[key] is None
            elif isinstance(a[key],(int,float,list)):
                np.testing.assert_allclose(a[key],b[key],rtol=0,atol=0,err_msg=key)
            else:assert a[key]==b[key],key


def compare_arrays(a,b,cases):
    for case in cases:
        with np.load(a/(case['name']+'.npz')) as left,np.load(b/(case['name']+'.npz')) as right:
            for key in left.files:np.testing.assert_array_equal(left[key],right[key],err_msg=case['name']+':'+key)


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);initial_source=source_hash()
    with tempfile.TemporaryDirectory(prefix='cospec_speed_check_') as tmp:
        root=Path(tmp);cfg=fixture(root/'fixture');meta=cfg['data']['PaviaU']
        old_batch=Patches.batch
        for patch in (9,15):
            for seed in (None,1,19):
                old=Patches(meta,'tr',patch,seed);new=Patches(meta,'tr',patch,seed)
                for index in ([0],[0,1],list(range(7)),list(range(13)),[23,0,15,3,22]):
                    a=old_batch(old,index);b=speed_data.vector_batch(new,index)
                    assert all(torch.equal(x,y) for x,y in zip(a,b))
                    if seed is not None:assert digest(old.rng.bit_generator.state)==digest(new.rng.bit_generator.state)
        print('[通过] 9/15窗口、边缘反射、旋转、奇数batch：像元/标签逐位一致，NumPy随机状态一致',flush=True)
        tr=np.load(Path(meta['cache'])/'tr.npy');tr.ravel()[22:24]=0
        np.save(Path(meta['cache'])/'tr.npy',tr);meta['train_count']=22
        job=dict(dataset='PaviaU',model='D',seed=2,patch=15,group='speed_pipeline_check')
        cfg['jobs']=[job];cfg['runtime_speedup']=dict(prefetch=True)
        def training(name,fast,stop_epoch=None,checkpoint=None):
            speed_runtime.uninstall()
            if fast:speed_runtime.install(prefetch=True)
            directory=root/name;directory.mkdir()
            if checkpoint:shutil.copy2(checkpoint,directory/'last.pt')
            seed_all(job['seed']);model=tiny_factory(cfg,job)
            prog=worker.Progress(directory/'progress.json',recipe_id(cfg,job))
            try:worker.train(model,cfg,job,directory,prog,lambda:False,stop_after_epoch=stop_epoch)
            except SafeStop:assert stop_epoch is not None
            return directory,model
        with (root/'train.log').open('w') as sink,contextlib.redirect_stdout(sink):
            baseline,bm=training('baseline',False)
            fast,fm=training('fast',True)
            old37,_=training('legacy37',False,37)
            resumed,rm=training('old_to_fast',True,checkpoint=old37/'last.pt')
            fast37,_=training('fast37',True,37)
            rollback,back=training('fast_to_old',False,checkpoint=fast37/'last.pt')
        for model in (fm,rm,back):assert state_hash(model.state_dict())==state_hash(bm.state_dict())
        saved=[torch.load(p/'last.pt',map_location='cpu',weights_only=False) for p in (baseline,fast,resumed,rollback)]
        for record in saved[1:]:
            assert digest(record['rotation_rng'])==digest(saved[0]['rotation_rng'])
            assert digest(record['mask_rng'])==digest(saved[0]['mask_rng'])
            assert torch.equal(record['loader_rng'],saved[0]['loader_rng'])
            assert torch.equal(record['rng']['torch'],saved[0]['rng']['torch'])
            for epoch in record['history']:
                for key in ('lr','loss','full_ce','missing_ce','last_gradient_norm','training_centers_seen'):
                    assert record['history'][epoch][key]==saved[0]['history'][epoch][key],key
        print('[通过] 固定0.001的100轮：新旧最终权重、损失与随机状态一致；旧→新、新→旧均可续训',flush=True)
        sha=state_hash(bm.state_dict());bm.eval()
        def evaluation(name,fast,stop=None):
            speed_runtime.uninstall()
            if fast:speed_runtime.install(prefetch=True)
            directory=root/name;directory.mkdir(exist_ok=True)
            progress=worker.Progress(directory/'progress.json',recipe_id(cfg,job))
            callback=(lambda:progress.values.get('test_steps_done',0)>=stop) if stop else (lambda:False)
            return directory,progress,callback
        a,ap,stop=evaluation('eval_baseline',False)
        metrics=worker.evaluate(bm,cfg,job,a,sha,ap,stop)
        b,bp,stop=evaluation('eval_fast',True)
        raw=speed_data.vector_batch;calls=[0]
        def counted(ds,indices):calls[0]+=1;return raw(ds,indices)
        speed_data.vector_batch=counted
        try:fast_metrics=worker.evaluate(bm,cfg,job,b,sha,bp,stop)
        finally:speed_data.vector_batch=raw
        compare_metrics(metrics,fast_metrics);compare_arrays(a,b,meta['cases'])
        assert calls[0]==4,calls
        print('[通过] 16条件的预测/置信度/全部指标逐项一致；读取图块从64批降为4批',flush=True)
        for name,start_fast,resume_fast,stop_steps in (
            ('legacy_cursor',False,True,1),('fast_cursor',True,False,7),
            ('fast_again',True,True,7),('last_pixel',True,True,49)):
            d,dp,stop=evaluation(name,start_fast,stop_steps)
            try:worker.evaluate(bm,cfg,job,d,sha,dp,stop)
            except SafeStop:pass
            else:raise AssertionError('evaluation did not stop')
            speed_runtime.uninstall()
            if resume_fast:speed_runtime.install(prefetch=True)
            out=worker.evaluate(bm,cfg,job,d,sha,dp,lambda:False)
            compare_metrics(metrics,out);compare_arrays(a,d,meta['cases'])
        print('[通过] 旧/新测试游标双向恢复、不同条件进度错开、最后一个像元中断，均无重计或漏测',flush=True)
        if args.models:
            cfg['train']['eval_batch']=2
            x,y=Patches(meta,'te',15).batch(range(2))
            full=torch.ones((2,32),dtype=torch.bool);missing=full.clone();missing[:,::2]=False
            for name in ('A','B','C','D','D_structure','D_row','D_raster4','D_random4','ConvMamba_aug',
                         'HybridSN','A2S2K','SQSFormer','GSCViT','MiM_v2'):
                seed_all(1);model=build_model(cfg,dict(job,model=name)).eval()
                with torch.inference_mode():
                    first=model(x,full)['logits'].clone()
                    model(x,missing)
                    second=model(x,full)['logits'].clone()
                assert torch.equal(first,second),name
                del model,first,second
                print('[通过] 实际模型eval切换掩码不污染后续预测：'+name,flush=True)
        speed_runtime.uninstall()
        assert source_hash()==initial_source
        print('[通过] 原科学代码哈希始终未改变',flush=True)
    print('加速兼容检查全部完成。')


if __name__=='__main__':main()
