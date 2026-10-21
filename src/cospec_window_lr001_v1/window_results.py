"""Join fresh 7/11 results with verified 9/15 JSON snapshots, never old 7/11."""
import csv
from pathlib import Path
import statistics
import shutil
import zipfile
from common import ROOT, job_dir, read_json, write_json
from results import complete
from window_protocol import parent_jobs


def csv_file(path, rows, fields):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def combined_table(config):
    root = Path(config['output'])
    parent = read_json(root / 'parent_snapshot/config.json')
    parent = dict(parent, output=str(root / 'parent_snapshot'))
    rows, groups = [], {}
    for cfg, jobs, source in ((config, config['jobs'], 'new_7_11'),
                              (parent, parent_jobs(parent), 'verified_9_15')):
        for job in jobs:
            if not complete(cfg, job):
                continue
            result = read_json(job_dir(cfg, job) / 'result.json')
            for fraction in (1., .8, .6, .4, .2, .1):
                masks = [m for m in result['metrics'] if m['fraction'] == fraction]
                row = dict(dataset=job['dataset'], model=job['model'], patch=job['patch'],
                           seed=job['seed'], fraction=fraction, source=source,
                           **{k: statistics.mean(m[k] for m in masks)
                              for k in ('oa_percent', 'aa_percent', 'kappa')})
                rows.append(row)
                groups.setdefault((job['patch'], job['model'], fraction), []).append(row)
    summary = []
    for patch in (7, 9, 11, 15):
        for model in ('B', 'D'):
            for fraction in (1., .8, .6, .4, .2, .1):
                seeds = groups.get((patch, model, fraction), [])
                row = dict(dataset='HongHu', patch=patch, model=model, fraction=fraction,
                           completed_seeds=len(seeds), expected_seeds=3, complete=len(seeds) == 3,
                           lr=.001, lr_policy='constant',
                           source='new_7_11' if patch in (7, 11) else 'verified_9_15')
                for metric in ('oa_percent', 'aa_percent', 'kappa'):
                    values = [s[metric] for s in seeds]
                    row[metric + '_mean'] = statistics.mean(values) if values else None
                    row[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else None
                summary.append(row)
    csv_file(root / 'window_comparison_by_seed.csv', rows,
             ['dataset', 'model', 'patch', 'seed', 'fraction', 'source', 'oa_percent', 'aa_percent', 'kappa'])
    csv_file(root / 'window_comparison.csv', summary,
             ['dataset', 'model', 'patch', 'fraction', 'completed_seeds', 'expected_seeds', 'complete',
              'lr', 'lr_policy', 'source', 'oa_percent_mean', 'oa_percent_std',
              'aa_percent_mean', 'aa_percent_std', 'kappa_mean', 'kappa_std'])
    table = []
    columns = ['patch', 'model', 'training_lr']
    for percent in (100, 80, 60, 40, 20, 10):
        columns.append(f'OA_keep{percent}_mean_sd')
    for patch in (7, 9, 11, 15):
        for model in ('B', 'D'):
            row = dict(patch=f'{patch}×{patch}', model=model, training_lr='constant 0.001')
            for percent in (100, 80, 60, 40, 20, 10):
                score = next(s for s in summary if
                             (s['patch'], s['model'], s['fraction']) == (patch, model, percent / 100))
                row[f'OA_keep{percent}_mean_sd'] = (f"{score['oa_percent_mean']:.2f} ± {score['oa_percent_std']:.2f}"
                    if score['complete'] else f"TBD ({score['completed_seeds']}/3)")
            table.append(row)
    csv_file(root / 'table_4_3.csv', table, columns)
    report = dict(complete=all(s['complete'] for s in summary), groups=summary,
                  note='Within each seed: average the three shared masks; then mean ± sample SD over three seeds. '
                       'Only newly trained 7/11 results and verified fixed-LR 9/15 snapshots are used.')
    write_json(root / 'window_comparison.json', report)
    return report


def package(config, tag='results'):
    from results import package as original_package
    target = original_package(config, tag)
    if tag == 'results':
        # The old packager filters by suffix and omits notebooks/LICENSEs even
        # though its frozen manifest verifies them. Preserve a restorable source.
        temporary = target.with_suffix('.zip.tmp')
        shutil.copyfile(target, temporary)
        with zipfile.ZipFile(temporary, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            existing = set(archive.namelist())
            for relative in read_json(ROOT / 'FROZEN_SHA256.json'):
                name = str(Path(Path(config['output']).name) / 'source/frozen' / relative)
                if name not in existing:
                    archive.write(ROOT / 'frozen' / relative, name)
        temporary.replace(target)
    return target
