"""Read-only access to the validated COSPEC_SUITE_V1 caches; no new split."""
from pathlib import Path
import hashlib
import numpy as np
from external_common import ROOT, read_json, write_json, file_hash

def mask_hash(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()

def validate_reference(reference, datasets, synthetic=False):
    reference = Path(reference).resolve()
    queue = read_json(reference / 'queue.json')
    if not synthetic:
        if read_json(reference / 'status.json').get('state') != 'completed':
            raise ValueError('旧99项队列尚未完成；请先完成并保留COSPEC_SUITE_V1目录')
        expected = read_json(ROOT / 'REFERENCE_IDENTITY.json')
        if queue['scope'] != 'formal' or queue['options']['patch'] != 15:
            raise ValueError('旧队列不是已核验的15×15正式实验')
    data = {}; errors = []
    for dataset in datasets:
        try:
            cache = reference / 'cache' / dataset
            meta = read_json(cache / 'data.json')
            if not synthetic:
                wanted = expected[dataset]
                if meta['data_id'] != wanted['data_id'] or meta['test_count'] != wanted['test_count']:
                    raise ValueError('数据身份/测试数量与已核验99项实验不一致')
                if [c['sha256'] for c in meta['cases']] != wanted['eval_cases']:
                    raise ValueError('测试掩码与已核验实验不一致')
            for name, sha in meta['cache_sha256'].items():
                if file_hash(cache / name) != sha: raise ValueError('缓存哈希不一致：' + name)
            cube = np.load(cache / 'cube.npy', mmap_mode='r')
            tr = np.load(cache / 'tr.npy', mmap_mode='r'); te = np.load(cache / 'te.npy', mmap_mode='r')
            if list(cube.shape) != meta['shape'] or tr.shape != te.shape or tr.shape != cube.shape[:2]:
                raise ValueError('影像/标签尺寸不一致')
            if np.any((tr > 0) & (te > 0)): raise ValueError('训练和测试中心重叠')
            if np.count_nonzero(tr) != meta['train_count'] or np.count_nonzero(te) != meta['test_count']:
                raise ValueError('训练/测试中心数不一致')
            if not synthetic:
                if np.bincount(te[te > 0], minlength=meta['classes'] + 1)[1:].tolist() != wanted['class_support']:
                    raise ValueError('测试类别数量不同')
            for case in meta['cases']:
                mask = np.asarray(case['mask'], bool)
                if len(mask) != cube.shape[-1] or mask_hash(mask) != case['sha256']:
                    raise ValueError('掩码内容或长度错误')
            for repeat in (1, 2, 3):
                masks = [np.asarray(next(c['mask'] for c in meta['cases'] if c['name'] == f'keep{f}_r{repeat}'), bool) for f in (40, 60, 80)]
                if not (np.all(masks[0] <= masks[1]) and np.all(masks[1] <= masks[2])):
                    raise ValueError('掩码不嵌套')
            data[dataset] = meta
            print(f'[{dataset}] 复用原划分 TR={meta["train_count"]}, TE={meta["test_count"]}, '
                  f'C={cube.shape[-1]}, K={meta["classes"]}；缓存哈希通过', flush=True)
        except Exception as error:
            errors.append(dict(dataset=dataset, error=str(error)))
            print(f'[数据检查失败] {dataset}: {error}', flush=True)
    return data, errors

class LazyPatches:
    def __init__(self, cache, part, patch=15, seed=None, coordinates=None):
        import torch
        self.cube = np.load(Path(cache) / 'cube.npy', mmap_mode='r')
        labels = np.load(Path(cache) / (part + '.npy'), mmap_mode='r')
        self.coords = np.argwhere(labels > 0) if coordinates is None else np.asarray(coordinates, np.int64)
        self.labels = labels[tuple(self.coords.T)].astype(np.int64) - 1
        self.rng = np.random.default_rng(seed) if seed is not None else None
        offsets = np.arange(-(patch // 2), patch // 2 + 1)
        self.rows = self.reflect(np.arange(self.cube.shape[0])[:, None] + offsets, self.cube.shape[0])
        self.cols = self.reflect(np.arange(self.cube.shape[1])[:, None] + offsets, self.cube.shape[1])
    @staticmethod
    def reflect(values, length):
        values = np.mod(values, 2 * (length - 1))
        return np.where(values < length, values, 2 * (length - 1) - values)
    def __len__(self): return len(self.coords)
    def __getitem__(self, index):
        import torch
        row, col = self.coords[index]
        x = self.cube[self.rows[row, :, None], self.cols[col, None, :], :]
        if self.rng is not None: x = np.rot90(x, int(self.rng.integers(4)), axes=(0, 1))
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), int(self.labels[index]), index

class ObservationSampler:
    def __init__(self, channels, seed, forbidden):
        self.channels = channels; self.rng = np.random.default_rng(seed)
        self.forbidden = {mask_hash(m) for m in forbidden}
    def draw(self, batch, device):
        import torch
        rows = []
        for _ in range(batch):
            for attempt in range(1000):
                keep = max(1, min(self.channels, round(self.channels * self.rng.uniform(.6, .9))))
                mask = np.zeros(self.channels, bool)
                mask[self.rng.choice(self.channels, keep, replace=False)] = True
                if mask.all(): mask[self.rng.integers(self.channels)] = False
                if mask_hash(mask) not in self.forbidden:
                    rows.append(mask); break
            else: raise RuntimeError('不能产生与测试方案不同的训练掩码')
        return torch.as_tensor(np.stack(rows), dtype=torch.bool, device=device)

def feature_coordinates(cache, per_class=50):
    labels = np.load(Path(cache) / 'te.npy'); rng = np.random.default_rng(10919); positions = []
    for cls in np.unique(labels[labels > 0]):
        coords = np.argwhere(labels == cls)
        positions.extend(coords[rng.choice(len(coords), min(per_class, len(coords)), replace=False)])
    return np.asarray(positions, dtype=np.int64)

def fit_pca(cache, destination, components=30):
    """Fit on complete training-center spectra only. Masking precedes transform."""
    cache = Path(cache); destination = Path(destination)
    cube = np.load(cache / 'cube.npy', mmap_mode='r'); tr = np.load(cache / 'tr.npy', mmap_mode='r')
    x = np.asarray(cube[tr > 0], np.float64)
    components = min(components, x.shape[1], len(x) - 1)
    if components < 13: raise ValueError('HybridSN的三层光谱卷积至少需要13个PCA分量')
    mean = x.mean(0); _, singular, vh = np.linalg.svd(x - mean, full_matrices=False)
    vectors = vh[:components].copy()
    signs = np.sign(vectors[np.arange(components), np.abs(vectors).argmax(1)])
    vectors *= signs[:, None]
    scale = singular[:components] / np.sqrt(len(x) - 1)
    if np.any(scale < 1e-10): raise ValueError('训练光谱PCA秩不足，拒绝以不稳定分量运行')
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, mean=mean.astype(np.float32), projection=(vectors / scale[:, None]).astype(np.float32))
    write_json(destination.with_suffix('.json'), dict(scope='training_center_spectra_only', training_centers=len(x),
        components=components, whiten=True, pca_sha256=file_hash(destination), mask_before_projection=True,
        means_sha256=hashlib.sha256(mean.tobytes()).hexdigest()))
    return dict(np.load(destination))
