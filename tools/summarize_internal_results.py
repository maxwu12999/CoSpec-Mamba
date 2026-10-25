#!/usr/bin/env python3
"""Rebuild manuscript Table 6 from the recorded per-mask metrics, without training."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('HongHu', 'PaviaU')
VARIANTS = ('B', 'LS', 'R', 'RDelta', 'D')
SEEDS = (1, 2, 3)
FRACTIONS = (1., .8, .6, .4, .2, .1)
MAIN = FRACTIONS[:4]
METRICS = ('oa_percent', 'aa_percent', 'kappa')
CONTRASTS = (('R-LS','R','LS'), ('RDelta-R','RDelta','R'),
             ('D-RDelta','D','RDelta'), ('D-R','D','R'), ('R-B','R','B'))


def write_csv(path, fields, rows):
    s = io.StringIO(newline='')
    w = csv.DictWriter(s, fields, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    path.write_text(s.getvalue(), encoding='utf-8')


def summarize(source, output):
    source, output = Path(source), Path(output)
    with source.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 480:
        raise ValueError('Expected 30 independent runs × 16 evaluated conditions')
    values, seen, shared_masks = {}, set(), {}
    for r in rows:
        d, v, seed, case = r['dataset'], r['variant'], int(r['seed']), r['case']
        frac = float(r['fraction'])
        if d not in DATASETS or v not in VARIANTS or seed not in SEEDS or frac not in FRACTIONS:
            raise ValueError('Unexpected run or retention fraction')
        key = (d, v, seed, case)
        if key in seen:
            raise ValueError('Duplicate run/case')
        seen.add(key)
        if frac == 1:
            if case != 'full': raise ValueError('Invalid full-spectrum case')
        elif case not in [f'keep{round(frac*100)}_r{i}' for i in (1,2,3)]:
            raise ValueError('Invalid nested-mask case name')
        mask_key = (d, case)
        if mask_key in shared_masks and shared_masks[mask_key] != r['mask_sha256']:
            raise ValueError('Test mask differs across models or training seeds')
        shared_masks[mask_key] = r['mask_sha256']
        for metric in METRICS:
            value = float(r[metric])
            if not (-1 <= value <= (1 if metric == 'kappa' else 100)):
                raise ValueError('Invalid accuracy or kappa')
            values.setdefault((d,v,seed,frac,metric), []).append(value)
    expected = {(d,v,s,f,m) for d in DATASETS for v in VARIANTS for s in SEEDS for f in FRACTIONS for m in METRICS}
    if set(values) != expected:
        raise ValueError('Missing run/fraction/metric')
    for key, xs in values.items():
        if len(xs) != (1 if key[3] == 1 else 3):
            raise ValueError('Average exactly three masks within each seed')
    means = {k: statistics.mean(v) for k,v in values.items()}
    per_seed = [dict(dataset=d,variant=v,seed=s,fraction=f,metric=m,value=x)
                for (d,v,s,f,m),x in sorted(means.items())]
    def fmt(xs):
        return f'{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}'
    table = []
    for d in DATASETS:
        for v in VARIANTS:
            row = dict(dataset=d,variant=v)
            for f in MAIN:
                row['full' if f == 1 else f'keep{round(f*100)}'] = fmt([means[d,v,s,f,'oa_percent'] for s in SEEDS])
            table.append(row)
    paired = []
    for d in DATASETS:
        for label,a,b in CONTRASTS:
            for f in MAIN:
                xs = [means[d,a,s,f,'oa_percent']-means[d,b,s,f,'oa_percent'] for s in SEEDS]
                paired.append(dict(dataset=d,contrast=label,fraction=f,paired_seeds=3,delta_OA_pp=fmt(xs)))
    output.mkdir(parents=True, exist_ok=True)
    fields = ['dataset','variant','full','keep80','keep60','keep40']
    write_csv(output/'table6_oa.csv', fields, table)
    write_csv(output/'per_seed.csv', ['dataset','variant','seed','fraction','metric','value'], per_seed)
    write_csv(output/'paired_differences.csv', ['dataset','contrast','fraction','paired_seeds','delta_OA_pp'], paired)
    md = ['# Internal spectral-encoding ablation', '',
          'OA (%), mean ± sample SD across three training seeds after averaging masks within each seed.',
          'RDelta denotes R+Δ. These runs are independent of the original main-experiment batch.', '',
          '| Dataset | Variant | Full | 80% | 60% | 40% |', '|---|---|---|---|---|---|']
    md += ['| '+' | '.join(r[k] for k in fields)+' |' for r in table]
    (output/'table6_oa.md').write_text('\n'.join(md)+'\n')
    (output/'validation.json').write_text(json.dumps(dict(training_runs=30,per_mask_records=len(rows),
        masks_averaged_within_seed=True,standard_deviation_ddof=1,
        manuscript_retention_fractions=list(MAIN),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n')
    return table


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'results/internal_v1/per_mask_metrics.csv')
    p.add_argument('--output', type=Path, default=ROOT/'results/internal_v1/summary')
    a = p.parse_args()
    summarize(a.input,a.output)
    print('Verified 30 runs and 480 conditions; wrote Table 6, per-seed metrics and paired differences.')
