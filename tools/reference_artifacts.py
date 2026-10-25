#!/usr/bin/env python3
"""Export exact paper split artifacts or rebuild a hash-verified reference cache.

Export requires the author's existing cache. It never generates a replacement
random split. Restore needs those artifacts and the original raw spectral cube.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_FILES = ('tr.npy', 'te.npy', 'wave.npy', 'stats.npz', 'data.json')
CACHE_FILES = ('cube.npy', 'tr.npy', 'te.npy', 'wave.npy', 'stats.npz')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def reference_identities():
    return json.loads((ROOT / 'src/cospec_full_lr001_v1/REFERENCE_DATA.json').read_text())


def validate(folder, expected, include_cube=True):
    folder = Path(folder)
    meta = json.loads((folder / 'data.json').read_text())
    if meta['data_id'] != expected['data_id'] or meta['shape'] != expected['shape']:
        raise ValueError('Reference identity/shape mismatch')
    for name in CACHE_FILES:
        if name == 'cube.npy' and not include_cube:
            continue
        actual = file_hash(folder / name)
        if actual != expected['cache_sha256'][name] or actual != meta['cache_sha256'][name]:
            raise ValueError(f'Original cache hash mismatch: {name}')
    tr = np.load(folder / 'tr.npy', allow_pickle=False)
    te = np.load(folder / 'te.npy', allow_pickle=False)
    wave = np.load(folder / 'wave.npy', allow_pickle=False)
    if tr.shape != tuple(meta['shape'][:2]) or te.shape != tr.shape:
        raise ValueError('Invalid split shape')
    if np.any((tr > 0) & (te > 0)):
        raise ValueError('Training and test centers overlap')
    if int(np.count_nonzero(tr)) != meta['train_count'] or int(np.count_nonzero(te)) != meta['test_count']:
        raise ValueError('Split support differs from paper')
    if wave.shape != (meta['shape'][2],) or not np.isfinite(wave).all() or not np.all(np.diff(wave) > 0):
        raise ValueError('Invalid original spectral coordinates')
    return meta


def export_reference(reference, output, datasets, identities=None):
    identities = reference_identities() if identities is None else identities
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    # Validate every input before writing anything.
    for name in datasets:
        validate(reference / 'cache' / name, identities[name])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reference-export-', dir=output.parent) as temp:
        stage = Path(temp) / 'artifacts'
        for name in datasets:
            target = stage / name
            target.mkdir(parents=True)
            for filename in ARTIFACT_FILES:
                shutil.copyfile(reference / 'cache' / name / filename, target / filename)
        manifest = {str(p.relative_to(stage)): file_hash(p) for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage / 'ARTIFACTS.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(output)
    return manifest


def restore_reference(artifacts, raw, output, dataset, mat_key=None, identities=None):
    identities = reference_identities() if identities is None else identities
    artifacts, output = Path(artifacts).resolve(), Path(output).resolve()
    manifest = json.loads((artifacts / 'ARTIFACTS.json').read_text())
    source = artifacts / dataset
    for name in ARTIFACT_FILES:
        if file_hash(source / name) != manifest[f'{dataset}/{name}']:
            raise ValueError(f'Export artifact corrupted: {dataset}/{name}')
    meta = validate(source, identities[dataset], include_cube=False)
    target = output / 'cache' / dataset
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite: {target}')
    raw = Path(raw)
    if raw.suffix.lower() == '.npy':
        cube = np.load(raw, allow_pickle=False)
    elif raw.suffix.lower() == '.mat' and mat_key:
        from scipy.io import loadmat
        cube = loadmat(raw)[mat_key]
    else:
        raise ValueError('Supply an H×W×C .npy or a non-HDF5 .mat with --mat-key')
    cube = np.asarray(cube, dtype=np.float32)
    if list(cube.shape) != meta['shape'] or not np.isfinite(cube).all():
        raise ValueError('Raw cube must preserve original shape and spectral band order')
    norm = meta['normalization']
    scale = float(norm['global_scale'])
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError('Invalid archived normalization scale')
    # Match the archived loader; keep published TR-only statistics unchanged.
    cube = ((cube - float(norm['global_mean'])) / scale).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reference-restore-', dir=target.parent) as temp:
        stage = Path(temp) / dataset
        stage.mkdir()
        for name in ARTIFACT_FILES:
            shutil.copyfile(source / name, stage / name)
        np.save(stage / 'cube.npy', cube)
        validate(stage, identities[dataset])
        stage.rename(target)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    e = commands.add_parser('export')
    e.add_argument('--reference', type=Path, required=True)
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--datasets', nargs='+', choices=tuple(reference_identities()), default=list(reference_identities()))
    r = commands.add_parser('restore')
    r.add_argument('--artifacts', type=Path, required=True)
    r.add_argument('--raw', type=Path, required=True)
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--dataset', choices=tuple(reference_identities()), required=True)
    r.add_argument('--mat-key')
    a = p.parse_args()
    if a.command == 'export':
        manifest = export_reference(a.reference, a.output, a.datasets)
        print(f'Exported {len(manifest)} exact split/coordinate/statistic files to {a.output}')
    else:
        print('Restored and verified:', restore_reference(a.artifacts, a.raw, a.output, a.dataset, a.mat_key))


if __name__ == '__main__':
    main()
