"""Average masks WITHIN seed, then mean/sample SD across three training seeds."""
import csv
import io
from pathlib import Path
import statistics
import bootstrap
from common import read_json, write_json, recipe_id, job_dir
from internal_models import VARIANTS, LABELS

FRACTIONS = (1., .8, .6, .4, .2, .1)
CONTRASTS = (('R-LS', 'R', 'LS'), ('RDelta-R', 'RDelta', 'R'),
             ('D-RDelta', 'D', 'RDelta'), ('D-R', 'D', 'R'), ('R-B', 'R', 'B'))


def write_csv(path, fields, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(stream.getvalue(), encoding='utf-8-sig')
    temporary.replace(path)


def collect(config):
    values, completed, missing = {}, [], []
    for job in config['jobs']:
        p = job_dir(config, job) / 'result.json'
        if not p.exists():
            missing.append(job); continue
        result = read_json(p)
        if (not result.get('complete') or result['recipe_id'] != recipe_id(config, job)
                or result['completed_epochs'] != config['train']['epochs']):
            raise ValueError(f'Incomplete/incompatible result: {p}')
        cases = config['data'][job['dataset']]['cases']
        metrics = result['metrics']
        if len(metrics) != len(cases) or len({m['case'] for m in metrics}) != len(cases):
            raise ValueError(f'Duplicated or missing test masks: {p}')
        expected = {c['name']: c for c in cases}
        for m in metrics:
            c = expected[m['case']]
            if (m['mask_sha256'] != c['sha256'] or m['recipe_id'] != result['recipe_id']
                    or m['model_sha256'] != result['model_sha256']
                    or m['count'] != config['data'][job['dataset']]['test_count']):
                raise ValueError(f'Wrong masks/model/test support: {p}')
        for frac in FRACTIONS:
            rows = [m for m in metrics if m['fraction'] == frac]
            if len(rows) != (1 if frac == 1 else 3):
                raise ValueError(f'Expected three masks per missing-band condition: {p}')
            for metric in ('oa_percent', 'aa_percent', 'kappa'):
                values[(job['dataset'], job['model'], job['seed'], frac, metric)] = statistics.mean(m[metric] for m in rows)
        completed.append(job)
    return values, completed, missing


def summarize(output):
    output = Path(output)
    config = read_json(output / 'config.json')
    values, completed, missing = collect(config)
    folder = output / 'summary'; folder.mkdir(exist_ok=True)
    rows = [dict(dataset=d, variant=v, seed=s, fraction=f, metric=m, value=x)
            for (d, v, s, f, m), x in sorted(values.items())]
    write_csv(folder / 'per_seed.csv', ['dataset', 'variant', 'seed', 'fraction', 'metric', 'value'], rows)
    # Keep the Word table order even while a selected subset is running.
    seeds = (1, 2, 3) if config['scope'] == 'formal' else tuple(config['seeds'])
    table, paired = [], []
    def stat(xs):
        if len(xs) != len(seeds):
            return 'TBD'
        return f'{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}' if len(xs) > 1 else f'{xs[0]:.2f} (1 seed)'
    for dataset in config['datasets']:
        for variant in VARIANTS:
            row = dict(dataset=dataset, variant=LABELS[variant])
            for frac in FRACTIONS:
                xs = [values[(dataset, variant, s, frac, 'oa_percent')] for s in seeds
                      if (dataset, variant, s, frac, 'oa_percent') in values]
                row['full' if frac == 1 else f'keep{round(100*frac)}'] = stat(xs)
            table.append(row)
        for label, a, b in CONTRASTS:
            for frac in FRACTIONS:
                diffs = [values[(dataset, a, s, frac, 'oa_percent')] - values[(dataset, b, s, frac, 'oa_percent')]
                         for s in seeds if (dataset, a, s, frac, 'oa_percent') in values
                         and (dataset, b, s, frac, 'oa_percent') in values]
                paired.append(dict(dataset=dataset, contrast=label, fraction=frac,
                                   paired_seeds=len(diffs), delta_OA_pp=stat(diffs)))
    fields = ['dataset', 'variant', 'full', 'keep80', 'keep60', 'keep40', 'keep20', 'keep10']
    write_csv(folder / 'table6_oa.csv', fields, table)
    write_csv(folder / 'paired_differences.csv', ['dataset', 'contrast', 'fraction', 'paired_seeds', 'delta_OA_pp'], paired)
    md = ['# 光谱编码内部消融', '', f'实验范围：{config["scope"]}；已完成 {len(completed)}/{len(config["jobs"])} 项。',
          '', '缺少三个训练种子时填写TBD；先对种子内三套掩码求均值，再计算三个种子的均值和样本标准差。', '',
          '| 数据集 | 变体 | 完整 | 80% | 60% | 40% |', '|---|---|---|---|---|---|']
    md += ['| ' + ' | '.join(str(r[k]) for k in fields[:6]) + ' |' for r in table]
    (folder / 'table6_oa.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    write_json(folder / 'completion.json', dict(scope=config['scope'], completed=len(completed),
               total=len(config['jobs']), missing=missing,
               formal_table_complete=(config['scope']=='formal' and set(config['datasets'])=={'HongHu','PaviaU'}
                    and all(all(r[k]!='TBD' for k in fields[2:]) for r in table))))
    return len(completed), len(config['jobs'])

