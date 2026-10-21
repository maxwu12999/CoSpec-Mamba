"""Read-only old caches. All 16 masks are shared across seeds and models."""
from pathlib import Path
import hashlib
import numpy as np
from common import ROOT, DATASETS, read_json, write_json, file_hash, digest


def mask_hash(mask):
    return hashlib.sha256(np.asarray(mask, np.uint8).tobytes()).hexdigest()


def all_cases(meta):
    old = {c['name']: c for c in meta['cases']}
    channels = meta['shape'][-1]
    full = dict(old['full'])
    if full['mask'] != [True] * channels or mask_hash(full['mask']) != full['sha256']:
        raise ValueError('完整波段掩码异常')
    result = [full]
    rng = np.random.default_rng(20261014)
    for repeat in (1, 2, 3):
        order = rng.permutation(channels)
        previous = np.ones(channels, bool)
        for fraction in (.8, .6, .4, .2, .1):
            mask = np.zeros(channels, bool)
            mask[order[:max(1, round(channels * fraction))]] = True
            name = f'keep{round(fraction * 100)}_r{repeat}'
            sha = mask_hash(mask)
            if not np.all(mask <= previous):
                raise ValueError('掩码未嵌套')
            previous = mask
            if fraction >= .4:
                if name not in old or old[name]['sha256'] != sha or old[name]['mask'] != mask.tolist():
                    raise ValueError('无法重建原排序：' + name)
                row = dict(old[name])
            else:
                row = dict(name=name, fraction=fraction, repeat=repeat, mask=mask.tolist(), sha256=sha,
                           actual_fraction=float(mask.mean()), retained_bands=int(mask.sum()),
                           parent_case=f'keep40_r{repeat}', parent_mask_sha256=old[f'keep40_r{repeat}']['sha256'])
            result.append(row)
    if len({c['sha256'] for c in result}) != 16:
        raise ValueError('测试掩码重复')
    return result


def validate_caches(reference, output, datasets=DATASETS, synthetic=False):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if reference == output or reference in output.parents or output in reference.parents:
        raise ValueError('新旧实验输出必须分开')
    expected = read_json(ROOT / 'REFERENCE_DATA.json')
    result = {}
    for dataset in datasets:
        cache = reference / 'cache' / dataset
        required = ('data.json', 'cube.npy', 'tr.npy', 'te.npy', 'stats.npz', 'wave.npy')
        for name in required:
            if not (cache / name).is_file():
                raise FileNotFoundError('缺少旧缓存：' + str(cache / name))
        meta = read_json(cache / 'data.json')
        wanted = expected[dataset] if not synthetic else meta
        if meta['data_id'] != wanted['data_id'] or meta['shape'] != wanted['shape']:
            raise ValueError(dataset + '数据身份与已归档实验不符')
        for name in required[1:]:
            sha = file_hash(cache / name)
            if sha != wanted['cache_sha256'][name] or sha != meta['cache_sha256'][name]:
                raise ValueError(dataset + '缓存校验失败：' + name)
        cube = np.load(cache / 'cube.npy', mmap_mode='r')
        tr = np.load(cache / 'tr.npy', mmap_mode='r')
        te = np.load(cache / 'te.npy', mmap_mode='r')
        wave = np.load(cache / 'wave.npy')
        if list(cube.shape) != meta['shape'] or tr.shape != cube.shape[:2] or te.shape != tr.shape:
            raise ValueError(dataset + '缓存尺寸异常')
        if np.any((tr > 0) & (te > 0)) or int((tr > 0).sum()) != meta['train_count'] or int((te > 0).sum()) != meta['test_count']:
            raise ValueError(dataset + '划分异常')
        support = np.bincount(te[te > 0], minlength=meta['classes'] + 1)[1:].tolist()
        if not synthetic:
            identity = read_json(ROOT / 'frozen/external/REFERENCE_IDENTITY.json')[dataset]
            if support != identity['class_support'] or meta['classes'] != identity['classes']:
                raise ValueError(dataset + '测试类分布异常')
        if not np.isfinite(wave).all() or wave.shape != (cube.shape[-1],) or not np.all(np.diff(wave) > 0):
            raise ValueError(dataset + '波长坐标无效')
        row = {k: meta[k] for k in ('dataset', 'shape', 'classes', 'train_count', 'test_count', 'data_id',
                                    'cache_sha256', 'normalization')}
        row.update(cases=all_cases(meta), class_support=support, cache=str(cache))
        result[dataset] = row
    return result


def prepare_pca(data, output):
    # Identical original training-center PCA, never fit with TE labels/pixels.
    from models import configure_imports
    configure_imports()
    from external_data import fit_pca
    result = {}
    for dataset, meta in data.items():
        path = Path(output) / 'preprocessing' / dataset / 'pca.npz'
        marker = path.with_suffix('.json')
        if path.exists():
            if not marker.exists() or file_hash(path) != read_json(marker)['pca_sha256']:
                raise ValueError('PCA文件已损坏：' + dataset)
        else:
            fit_pca(meta['cache'], path, 30)
        result[dataset] = dict(path=str(path), sha256=file_hash(path))
    return result


class Patches:
    def __init__(self, meta, part, patch, seed=None, coordinates=None):
        cache = Path(meta['cache'])
        self.cube = np.load(cache / 'cube.npy', mmap_mode='r')
        labels = np.load(cache / (part + '.npy'), mmap_mode='r')
        self.coords = np.argwhere(labels > 0) if coordinates is None else np.asarray(coordinates, np.int64)
        self.labels = np.asarray(labels[tuple(self.coords.T)], np.int64) - 1
        self.rng = np.random.default_rng(seed) if seed is not None else None
        offset = np.arange(-(patch // 2), patch // 2 + 1)
        self.rows = self.reflect(np.arange(self.cube.shape[0])[:, None] + offset, self.cube.shape[0])
        self.cols = self.reflect(np.arange(self.cube.shape[1])[:, None] + offset, self.cube.shape[1])

    @staticmethod
    def reflect(values, size):
        x = np.mod(values, 2 * (size - 1))
        return np.where(x < size, x, 2 * (size - 1) - x)

    def __len__(self):
        return len(self.coords)

    def batch(self, indices):
        import torch
        xs, ys = [], []
        for i in indices:
            r, c = self.coords[i]
            x = self.cube[self.rows[r, :, None], self.cols[c, None, :], :]
            if self.rng is not None:
                x = np.rot90(x, int(self.rng.integers(4)), axes=(0, 1))
            xs.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
            ys.append(int(self.labels[i]))
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def feature_coordinates(meta, per_class=50):
    labels = np.load(Path(meta['cache']) / 'te.npy', mmap_mode='r')
    rng = np.random.default_rng(10919)
    parts = []
    for cls in range(1, meta['classes'] + 1):
        co = np.argwhere(labels == cls)
        parts.append(co[rng.choice(len(co), min(per_class, len(co)), replace=False)])
    return np.concatenate(parts)
