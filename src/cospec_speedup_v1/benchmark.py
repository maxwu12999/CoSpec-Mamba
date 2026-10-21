#!/usr/bin/env python3
"""Small isolated CUDA correctness/throughput check. Never writes formal weights."""
from bootstrap import use_core,revision
use_core()
import argparse
import copy
from pathlib import Path
import signal
import tempfile
import time
import numpy as np
import torch
from common import read_json,write_json,digest,recipe_id,SafeStop
from data import Patches,feature_coordinates
from models import build_model,configure_imports
configure_imports()
from external_common import seed_all,state_hash
import worker
import speed_data,speed_runtime


def subset_config(original,root,dataset,points):
    config=copy.deepcopy(original);meta=config['data'][dataset]
    cache=root/'cache'/dataset;cache.mkdir(parents=True)
    source=Path(meta['cache'])
    for name in ('cube.npy','tr.npy','wave.npy','stats.npz'):(cache/name).symlink_to((source/name).resolve())
    truth=np.load(source/'te.npy',mmap_mode='r');coords=feature_coordinates(meta,per_class=max(1,(points+meta['classes']-1)//meta['classes']))[:points]
    labels=np.zeros_like(truth);labels[tuple(coords.T)]=truth[tuple(coords.T)]
    np.save(cache/'te.npy',labels)
    meta.update(cache=str(cache),test_count=len(coords),class_support=np.bincount(labels[labels>0],minlength=meta['classes']+1)[1:].tolist())
    config.update(output=str(root/'temporary_probe_output'),scope='runtime_benchmark_only',
                  protocol_id=digest(dict(original=original['protocol_id'],revision=revision(),test_coords=coords.tolist())),
                  runtime_speedup=dict(prefetch=original.get('runtime_speedup',{}).get('prefetch',True)))
    return config


def equivalent(left,right,left_dir,right_dir,cases):
    for a,b in zip(left,right):
        assert a['case']==b['case']
        for key in ('confusion_matrix','class_support','count'):np.testing.assert_array_equal(a[key],b[key])
        for key in ('oa_percent','aa_percent','kappa','nll','brier','ece_15_bins','high_confidence_error_fraction'):
            if a[key] is None:assert b[key] is None
            else:np.testing.assert_allclose(a[key],b[key],rtol=1e-5,atol=1e-6,err_msg=key)
    for case in cases:
        with np.load(left_dir/(case['name']+'.npz')) as a,np.load(right_dir/(case['name']+'.npz')) as b:
            for key in ('positions','labels','predictions','observed'):np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_allclose(a['confidence'],b['confidence'],rtol=1e-5,atol=1e-6)


def run(original,report,device='cuda',points=1024,models=('D','ConvMamba_aug'),factory=None):
    torch.set_num_threads(1)
    stopped=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    stop=lambda:stopped[0] or (Path(original['output'])/'STOP').exists()
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA不可用')
    rows=[]
    dataset='HongHu' if 'HongHu' in original['data'] else next(iter(original['data']))
    with tempfile.TemporaryDirectory(prefix='cospec_speed_gpu_probe_') as temporary:
        root=Path(temporary);config=subset_config(original,root,dataset,points);config['device']=device
        for name in models:
            if stop():raise SafeStop()
            job=dict(dataset=dataset,model=name,seed=1,patch=15,group='runtime_benchmark_only')
            speed_runtime.uninstall();seed_all(1)
            model=(factory or build_model)(config,job).to(device).eval();sha=state_hash(model.state_dict())
            meta=config['data'][dataset]
            # Validate raw training inputs and rotation state before attempting prefetch.
            old=Patches(meta,'tr',15,12);new=Patches(meta,'tr',15,12)
            for start in (0,32,64):
                if start>=len(old):continue
                index=list(range(start,min(start+32,len(old))))
                a=old.batch(index);b=speed_data.vector_batch(new,index)
                assert all(torch.equal(x,y) for x,y in zip(a,b))
                assert digest(old.rng.bit_generator.state)==digest(new.rng.bit_generator.state)
            warm=Patches(meta,'te',15).batch(range(min(config['train']['eval_batch'],meta['test_count'])))[0].to(device)
            mask=torch.ones((len(warm),meta['shape'][-1]),dtype=torch.bool,device=device)
            with torch.inference_mode():
                for _ in range(3):model(warm,mask)
            if device=='cuda':torch.cuda.synchronize()
            del warm,mask
            timings={False:[],True:[]};peaks={False:[],True:[]};reference=None;reference_dir=None
            for run_index,fast in enumerate((False,True,True,False)):
                if stop():raise SafeStop()
                speed_runtime.uninstall()
                if fast:speed_runtime.install(prefetch=config['runtime_speedup']['prefetch'])
                directory=root/(name+str(run_index));directory.mkdir()
                progress=worker.Progress(directory/'progress.json',recipe_id(config,job))
                if device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                began=time.perf_counter()
                metrics=worker.evaluate(model,config,job,directory,sha,progress,stop)
                if device=='cuda':torch.cuda.synchronize()
                elapsed=time.perf_counter()-began
                timings[fast].append(elapsed)
                peaks[fast].append(torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else 0.)
                if reference is None:reference,reference_dir=metrics,directory
                else:equivalent(reference,metrics,reference_dir,directory,meta['cases'])
                assert state_hash(model.state_dict())==sha
            legacy=float(np.median(timings[False]));fast=float(np.median(timings[True]))
            row=dict(dataset=dataset,model=name,test_pixels=meta['test_count'],cases=16,
                     legacy_seconds=legacy,speed_seconds=fast,speedup_ratio=legacy/fast,
                     legacy_peak_mib=max(peaks[False]),speed_peak_mib=max(peaks[True]),
                     predictions_equal=True,training_inputs_and_rotation_state_equal=True)
            rows.append(row)
            print(f'[测速] {name}：原版{legacy:.2f}s，加速版{fast:.2f}s，倍率{legacy/fast:.2f}；16条件预测与指标通过',flush=True)
            del model
            if device=='cuda':torch.cuda.empty_cache()
        speed_runtime.uninstall()
    ratio=sum(r['legacy_seconds'] for r in rows)/sum(r['speed_seconds'] for r in rows)
    fits=all(r['speed_peak_mib']<=r['legacy_peak_mib']*1.10+128 for r in rows)
    result=dict(passed=True,enabled=ratio>=1.02 and fits,revision=revision(),protocol_id=original['protocol_id'],
                device=device,torch=str(torch.__version__),rows=rows,weighted_test_speedup=ratio,
                reason='enable' if ratio>=1.02 and fits else 'sample did not show enough benefit or memory grew; use original worker',
                note='isolated subset benchmark; not a claim about full-queue speed or model accuracy')
    write_json(report,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--points',type=int,default=1024);a=p.parse_args()
    try:run(read_json(a.config),a.report,points=a.points)
    except SafeStop:return 130
    return 0


if __name__=='__main__':raise SystemExit(main())
