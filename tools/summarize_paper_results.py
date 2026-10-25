#!/usr/bin/env python3
"""Rebuild main, window, encoding and scan tables from archived per-case CSVs."""
import argparse
import csv
import io
import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('HongHu','HanChuan','PaviaU','LongKou','Houston','IndianPines','Salinas')
MODELS = ('ConvMamba_aug','HybridSN','A2S2K','SQSFormer','GSCViT','MiM_v2','D')
MAIN = (1.,.8,.6,.4)
ALL = MAIN + (.2,.1)
FIELDS = ('full','keep80','keep60','keep40')


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def write(path,fields,rows):
    s=io.StringIO(newline='')
    w=csv.DictWriter(s,fields,lineterminator='\n')
    w.writeheader();w.writerows(rows)
    path.write_text(s.getvalue())


def summarize(root=ROOT,output=None):
    root=Path(root);output=Path(output) if output else root/'results/paper_tables'
    full=read(root/'results/full_lr001_v1/metrics_raw.csv')
    window=read(root/'results/window_lr001_v1/metrics_raw.csv')
    if len(full)!=3360 or len(window)!=192:
        raise ValueError('Expected 210 original runs and 12 additional window runs')
    seen=set();groups={}
    for r in full+window:
        d,m,p,s,f=r['dataset'],r['model'],int(r['patch']),int(r['seed']),float(r['fraction'])
        key=(d,m,p,s,r['case'])
        if key in seen:raise ValueError('Duplicate training run/case; do not recount reused window snapshots')
        seen.add(key)
        if s not in (1,2,3) or f not in ALL:raise ValueError('Invalid seed or retention fraction')
        if r['case'] not in (['full'] if f==1 else [f'keep{round(f*100)}_r{i}' for i in (1,2,3)]):
            raise ValueError('Invalid case name')
        groups.setdefault((d,m,p,s,f),[]).append(float(r['oa_percent']))
    means={}
    for k,v in groups.items():
        if len(v)!=(1 if k[-1]==1 else 3):raise ValueError('Incomplete within-seed mask set')
        means[k]=st.mean(v)
    def xs(d,m,p,f):
        return [means[d,m,p,s,f] for s in (1,2,3)]
    def average(d,m,p,f):return st.mean(xs(d,m,p,f))
    def fmt(v,signed=False):
        return f'{st.mean(v):+.2f} ± {st.stdev(v):.2f}' if signed else f'{st.mean(v):.2f} ± {st.stdev(v):.2f}'
    def four(d,m,p):return dict(zip(FIELDS,[fmt(xs(d,m,p,f)) for f in MAIN]))
    output.mkdir(parents=True,exist_ok=True)
    window_rows=[dict(patch=p,variant=m,**four('HongHu',m,p)) for p in (7,9,11,15) for m in ('B','D')]
    write(output/'table3_window.csv',['patch','variant',*FIELDS],window_rows)
    encoding=[dict(dataset=d,variant=m,**four(d,m,15)) for d in ('HongHu','PaviaU') for m in ('A','B','C','D')]
    write(output/'table4_encoding_augmentation.csv',['dataset','variant',*FIELDS],encoding)
    differences=[]
    for d in DATASETS:
        row=dict(dataset=d)
        for field,f in zip(FIELDS,MAIN):
            row[field]=fmt([a-b for a,b in zip(xs(d,'D',15,f),xs(d,'B',15,f))],signed=True)
        differences.append(row)
    differences.append(dict(dataset='Unweighted mean difference',**{
        field:f'{st.mean(average(d,"D",15,f)-average(d,"B",15,f) for d in DATASETS):+.2f}' for field,f in zip(FIELDS,MAIN)}))
    write(output/'table5_paired_d_minus_b.csv',['dataset',*FIELDS],differences)
    scan=[dict(dataset=d,variant=m,**four(d,m,15)) for d in ('HongHu','PaviaU')
          for m in ('D_row','D_raster4','D_random4','D','D_structure')]
    write(output/'table7_scan.csv',['dataset','variant',*FIELDS],scan)
    for suffix,f in zip(('8a_full','8b_keep80','8c_keep60','8d_keep40'),MAIN):
        rows=[dict(dataset=d,**{m:fmt(xs(d,m,15,f)) for m in MODELS}) for d in DATASETS]
        rows.append(dict(dataset='Unweighted mean',**{m:f'{st.mean(average(d,m,15,f) for d in DATASETS):.2f}' for m in MODELS}))
        write(output/f'table{suffix}.csv',['dataset',*MODELS],rows)
    low=[dict(dataset=d,fraction=f,**{m:fmt(xs(d,m,15,f)) for m in ('A2S2K','SQSFormer','D')})
         for d in ('HongHu','PaviaU') for f in (.2,.1)]
    write(output/'table8e_low_retention.csv',['dataset','fraction','A2S2K','SQSFormer','D'],low)
    overall=[]
    for f in MAIN:
        b=st.mean(average(d,'B',15,f) for d in DATASETS)
        a=st.mean(average(d,'D',15,f) for d in DATASETS)
        overall.append(dict(fraction=f,B_mean_OA=f'{b:.2f}',D_mean_OA=f'{a:.2f}',
                            delta_OA_pp=f'{a-b:+.2f}',datasets_favoring_D=f'{sum(average(d,"D",15,f)>average(d,"B",15,f) for d in DATASETS)}/7'))
    write(output/'table12_bd_overall.csv',['fraction','B_mean_OA','D_mean_OA','delta_OA_pp','datasets_favoring_D'],overall)
    trend=[dict(dataset=d,model=m,fraction=f,OA_mean=average(d,m,15,f)) for d in DATASETS for m in MODELS for f in ALL]
    write(output/'external_trends.csv',['dataset','model','fraction','OA_mean'],trend)
    report=dict(main_training_runs=210,new_window_training_runs=12,unique_training_runs=222,
        evaluated_conditions=len(seen),epochs=100,standard_deviation_ddof=1,
        aggregation='Average masks within each seed, then mean/sample SD across three seeds; paired differences computed before across-seed aggregation.',
        separate_internal_batch='results/internal_v1 is not merged into the original main comparisons')
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    a=p.parse_args();print(json.dumps(summarize(output=a.output),indent=2))
