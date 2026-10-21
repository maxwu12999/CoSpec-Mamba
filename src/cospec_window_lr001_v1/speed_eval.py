"""Reuse one CPU/GPU input batch across cases; retain per-case old cursor format."""
import contextlib
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from common import read_json,write_json,recipe_id,SafeStop
from data import Patches
from speed_data import prepare_batches


def evaluate(model,config,job,directory,model_sha,progress,stop):
    from worker import Metrics,atomic_npz
    dataset=Patches(config['data'][job['dataset']],'te',job['patch'])
    meta=config['data'][job['dataset']]
    batch=config['train']['eval_batch'];n=len(dataset)
    steps=(n+batch-1)//batch;rid=recipe_id(config,job)
    finished={};states=[]
    prefetch=config.get('runtime_speedup',{}).get('prefetch',True)

    def save(state):
        state['pred'].flush();state['conf'].flush()
        write_json(state['marker'],dict(**state['identity'],next_pixel=state['next'],metrics=state['acc'].s))
        state['dirty']=False

    def save_all():
        for state in states:
            if state['case']['name'] not in finished and state['dirty']:save(state)

    def finish(state):
        began=time.perf_counter();case=state['case']
        atomic_npz(state['prediction'],labels=dataset.labels,predictions=np.asarray(state['pred']),
                   confidence=np.asarray(state['conf']),positions=dataset.coords,observed=np.asarray(case['mask'],bool))
        row=dict(**state['identity'],case=case['name'],fraction=case['fraction'],repeat=case['repeat'],
                 seconds_this_segment=state['seconds']+time.perf_counter()-began,
                 timing_scope='shared input preparation apportioned across cases; concurrent, not isolated latency',
                 **state['acc'].finish())
        if row['count']!=meta['test_count'] or row['class_support']!=meta['class_support']:
            raise ValueError('测试像元未完整覆盖')
        write_json(state['metric_file'],row);finished[case['name']]=row
        state['pred']=state['conf']=None
        shutil.rmtree(state['partial'])

    def done_steps():
        return len(finished)*steps+sum((s['next']+batch-1)//batch for s in states if s['case']['name'] not in finished)

    for case in meta['cases']:
        prediction=directory/(case['name']+'.npz');metric_file=directory/(case['name']+'.metrics.json')
        if prediction.exists() and metric_file.exists():
            row=read_json(metric_file)
            if (row['recipe_id'],row['model_sha256'],row['mask_sha256'],row['count'])!=(rid,model_sha,case['sha256'],n):
                raise ValueError('已有测试结果协议不符')
            finished[case['name']]=row;continue
        partial=directory/'partial'/case['name'];partial.mkdir(parents=True,exist_ok=True)
        marker=partial/'cursor.json'
        identity=dict(recipe_id=rid,model_sha256=model_sha,mask_sha256=case['sha256'])
        cursor=read_json(marker) if marker.exists() else None
        if cursor and any(cursor.get(k)!=v for k,v in identity.items()):
            raise ValueError('测试断点权重或掩码不匹配')
        offset=cursor['next_pixel'] if cursor else 0
        acc=Metrics(meta['classes'],cursor['metrics'] if cursor else None)
        if acc.s['count']!=offset or not 0<=offset<=n or (offset!=n and offset%batch):
            raise ValueError('测试游标和统计数量异常')
        mode='r+' if cursor else 'w+'
        pred=np.lib.format.open_memmap(partial/'pred.npy',mode=mode,dtype=np.int16,shape=(n,))
        conf=np.lib.format.open_memmap(partial/'confidence.npy',mode=mode,dtype=np.float32,shape=(n,))
        if pred.shape!=(n,) or conf.shape!=(n,):raise ValueError('测试断点尺寸异常')
        state=dict(case=case,prediction=prediction,metric_file=metric_file,partial=partial,marker=marker,
                   identity=identity,next=offset,acc=acc,pred=pred,conf=conf,dirty=cursor is None,seconds=0.)
        states.append(state)
        if offset==n:finish(state)
    pending=[s for s in states if s['case']['name'] not in finished]
    if not pending:return [finished[c['name']] for c in meta['cases']]
    first=min(s['next'] for s in pending)
    parts=(range(offset,min(offset+batch,n)) for offset in range(first,n,batch))
    model.eval()
    progress.update(force=True,phase='test',test_steps_done=done_steps(),test_steps_total=len(meta['cases'])*steps,
                    test_pipeline='one input batch reused across unfinished masks')
    started=time.perf_counter();loaded=0
    try:
        with torch.inference_mode(),contextlib.closing(prepare_batches(dataset,parts,prefetch)) as prepared:
            for indices in prepared:
                if stop():save_all();raise SafeStop()
                offset=indices[0];end=indices[-1]+1
                need=[s for s in pending if s['case']['name'] not in finished and s['next']==offset]
                if not need:raise ValueError('测试游标跨批次不一致')
                began=time.perf_counter()
                x,y=dataset.batch(indices)
                x_device=x.to(config['device'])
                common_time=time.perf_counter()-began;loaded+=1
                for state in need:
                    began=time.perf_counter();case=state['case']
                    mask=torch.tensor(case['mask'],dtype=torch.bool,device=config['device'])
                    logits=model(x_device,mask[None].expand(len(y),-1))['logits']
                    if not torch.isfinite(logits).all():raise FloatingPointError('测试logits包含NaN/Inf')
                    prob=logits.softmax(-1).cpu().numpy()
                    pred,conf=state['acc'].add(y.numpy(),prob)
                    state['pred'][offset:end]=pred;state['conf'][offset:end]=conf
                    state['next']=end;state['dirty']=True
                    state['seconds']+=common_time/len(need)+time.perf_counter()-began
                    del logits,prob,pred,conf,mask
                    progress.update(phase='test',case=case['name'],test_steps_done=done_steps(),
                                    test_steps_total=len(meta['cases'])*steps,current_case_pixel=end,current_case_total=n)
                    want_stop=stop()
                    if want_stop:save_all();raise SafeStop()
                    if end==n:finish(state)
                    elif ((end+batch-1)//batch)%50==0:save(state)
                del x_device,x,y
        save_all()
    finally:
        # On any exception, only already-committed cursors are trusted on retry.
        # Dirty mmap rows may be overwritten from that cursor without double-counting.
        elapsed=time.perf_counter()-started
        write_json(directory/'speedup_test_progress.json',dict(input_batches_prepared=loaded,
                   seconds_this_segment=elapsed,finished_cases=list(finished),input_reuse=True))
    progress.update(force=True,phase='test',test_steps_done=len(meta['cases'])*steps)
    if config['device'].startswith('cuda'):torch.cuda.empty_cache()
    return [finished[c['name']] for c in meta['cases']]
