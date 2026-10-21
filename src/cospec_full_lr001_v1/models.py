"""Unmodified archived model definitions; one adapter for all 14 model variants."""
from pathlib import Path
import sys
from common import ROOT, EXTERNAL


def configure_imports():
    for part in ('suite', 'external'):
        value = str(ROOT / 'frozen' / part)
        if value not in sys.path:
            sys.path.insert(0, value)


def build_model(config, job):
    import numpy as np
    configure_imports()
    meta = config['data'][job['dataset']]
    cache = Path(meta['cache'])
    if job['model'] in EXTERNAL:
        from external_models import ExternalModel
        pca = dict(np.load(config['pca'][job['dataset']]['path'])) if job['model'] == 'HybridSN' else None
        return ExternalModel(job['model'], job['dataset'], meta['shape'][-1], meta['classes'],
                             patch=job['patch'], pca=pca, scan_backend=config['backend']['external'])
    from suite_models import build_model as build
    return build(job['model'], np.load(cache / 'wave.npy'), dict(np.load(cache / 'stats.npz')), job['seed'],
                 dict(patch=job['patch'], dim=64, depth=4, basis=64, backend=config['backend']['suite']))


def augmentation(job):
    return job['model'] not in ('A', 'C')
