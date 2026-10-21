"""Seed-aware statistics; masks are never counted as independent training seeds."""
import csv
from pathlib import Path
import statistics
import zipfile
from common import ROOT, read_json, write_json, job_dir, recipe_id


def complete(config, job):
    path = job_dir(config, job) / 'result.json'
    if not path.exists():
        return False
    result = read_json(path)
    if result['recipe_id'] != recipe_id(config, job):
        raise ValueError('完成结果协议不同：' + str(path))
    if not result.get('complete'):
        return False
    if len(result.get('metrics', [])) != 16 or result.get('completed_epochs') != config['train']['epochs']:
        raise ValueError('完成结果未覆盖完整轮次/16测试条件')
    if result.get('job') != job or result.get('lr_policy') != 'constant_0.001':
        raise ValueError('完成结果的任务或实际学习率标识异常')
    cases = {c['name']: c for c in config['data'][job['dataset']]['cases']}
    names = [m['case'] for m in result['metrics']]
    if len(set(names)) != 16 or set(names) != set(cases):
        raise ValueError('完成结果包含重复或缺失的测试条件')
    for m in result['metrics']:
        expected = cases[m['case']]
        if (m['mask_sha256'] != expected['sha256'] or
                m['fraction'] != expected['fraction'] or m['repeat'] != expected['repeat'] or
                m['count'] != config['data'][job['dataset']]['test_count'] or
                m['class_support'] != config['data'][job['dataset']]['class_support']):
            raise ValueError('完成结果掩码/测试数量不符')
    return True


def aggregate(config):
    root = Path(config['output'])
    raw, seed_rows, groups = [], [], {}
    completed = 0
    for job in config['jobs']:
        if not complete(config, job):
            continue
        completed += 1
        result = read_json(job_dir(config, job) / 'result.json')
        for row in result['metrics']:
            raw.append(dict(dataset=job['dataset'], model=job['model'], patch=job['patch'], seed=job['seed'],
                            case=row['case'], fraction=row['fraction'], oa_percent=row['oa_percent'],
                            aa_percent=row['aa_percent'], kappa=row['kappa'], count=row['count']))
        for fraction in (1., .8, .6, .4, .2, .1):
            metrics = [m for m in result['metrics'] if m['fraction'] == fraction]
            if len(metrics) != (1 if fraction == 1 else 3):
                raise ValueError('保留率测试不完整')
            row = dict(dataset=job['dataset'], model=job['model'], patch=job['patch'], seed=job['seed'], fraction=fraction,
                       **{k: statistics.mean(m[k] for m in metrics) for k in ('oa_percent','aa_percent','kappa')})
            seed_rows.append(row)
            groups.setdefault((job['dataset'], job['model'], job['patch'], fraction), []).append(row)
    summary = []
    for (dataset, model, patch, fraction), values in sorted(groups.items()):
        row = dict(dataset=dataset, model=model, patch=patch, fraction=fraction,
                   completed_seeds=len(values), expected_seeds=3,
                   complete=len(values)==3)
        for metric in ('oa_percent', 'aa_percent', 'kappa'):
            scores = [v[metric] for v in values]
            row[metric+'_mean'] = statistics.mean(scores)
            row[metric+'_std'] = statistics.stdev(scores) if len(scores)>1 else None
        summary.append(row)
    for filename, rows, columns in [
        ('metrics_raw.csv', raw, ['dataset','model','patch','seed','case','fraction','oa_percent','aa_percent','kappa','count']),
        ('metrics_by_seed.csv', seed_rows, ['dataset','model','patch','seed','fraction','oa_percent','aa_percent','kappa']),
        ('summary.csv', summary, ['dataset','model','patch','fraction','completed_seeds','expected_seeds','complete',
         'oa_percent_mean','oa_percent_std','aa_percent_mean','aa_percent_std','kappa_mean','kappa_std'])]:
        temp = root / (filename + '.tmp')
        with temp.open('w', encoding='utf-8-sig', newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        temp.replace(root / filename)
    report = dict(complete=completed==len(config['jobs']), completed=completed, total=len(config['jobs']),
                  training_lr='constant_0.001', rates=[1.,.8,.6,.4,.2,.1],
                  aggregation='mean3 masks inside each training seed, then mean and sample std over3 training seeds',
                  groups=summary)
    write_json(root/'summary.json', report)
    return report


def package(config, tag='results'):
    root = Path(config['output'])
    target = root.with_name(root.name + '_' + tag + '.zip')
    temporary = target.with_name(target.name + '.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix in ('.pt','.tmp','.lock') or 'partial' in path.parts:
                continue
            relative = path.relative_to(root)
            selected = (tag=='maps' and relative.parts[0]=='maps') or (tag=='features' and relative.parts[0]=='features')
            selected |= tag=='results' and relative.parts[0] not in ('maps','features','profiles') and path.suffix in ('.json','.csv','.log','.txt')
            # Detailed prediction arrays are retained in the server's runs, not
            # copied into the small download. Split masks are in old cache.
            if selected:
                z.write(path,str(Path(root.name)/relative))
        if tag=='results':
            for path in sorted(ROOT.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.sh','.json','.md','.txt'):
                    z.write(path,str(Path(root.name)/'source'/path.relative_to(ROOT)))
            for path in sorted((root/'profiles').glob('*.json')):
                z.write(path,str(Path(root.name)/'profiles'/path.name))
    temporary.replace(target)
    return target
