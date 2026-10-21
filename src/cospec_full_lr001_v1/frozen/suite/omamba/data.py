"""Fixed TR/TE protocol, training-only statistics, raw spectral channels."""
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .metadata import file_hash


def array_hash(array):
    import hashlib
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def load_split(path):
    path = Path(path).expanduser().resolve()
    obj = loadmat(path)
    if not all(k in obj for k in ("input", "TR", "TE")):
        raise ValueError("需要包含 input、TR、TE 的 split.mat；原生 HanChuan 可先运行 prepare_hanchuan.py。")
    cube = np.asarray(obj["input"], dtype=np.float32)
    tr = np.asarray(obj["TR"])
    te = np.asarray(obj["TE"])
    if cube.ndim != 3 or tr.shape != cube.shape[:2] or te.shape != tr.shape:
        raise ValueError("影像必须是 H×W×C，TR/TE 必须是同尺寸二维标签掩码。")
    if not all(np.all(np.isfinite(x)) for x in (cube, tr, te)):
        raise ValueError("数据含 NaN/Inf。")
    for labels in (tr, te):
        if np.any(labels < 0) or np.any(labels != np.round(labels)):
            raise ValueError("标签必须是 0 背景、1..C 的整数。")
    tr, te = tr.astype(np.int64), te.astype(np.int64)
    if np.any((tr > 0) & (te > 0)):
        raise ValueError("TR 和 TE 中心像素存在重叠。")
    train_coords, test_coords = np.argwhere(tr > 0), np.argwhere(te > 0)
    if not len(train_coords) or not len(test_coords):
        raise ValueError("训练或测试集合为空。")
    classes = np.unique(tr[tr > 0])
    if not np.array_equal(classes, np.arange(1, len(classes) + 1)):
        raise ValueError("训练类别必须覆盖连续的 1..C。")
    if np.any(te > len(classes)):
        raise ValueError("TE 包含训练集合未覆盖的类别。")
    # A single global affine transform preserves band mixing and spectral shape.
    # No scene-wide fitting, no PCA, no per-band independently fitted scaling.
    train_spectra = cube[tr > 0].astype(np.float64)
    mean, scale = float(train_spectra.mean()), float(train_spectra.std())
    if scale < 1e-8:
        raise ValueError("训练光谱没有有效变化。")
    normalized = (cube - mean) / scale
    spectra = normalized[tr > 0]
    train_labels = tr[tr > 0] - 1
    means, variances, counts = [], [], []
    pooled = np.maximum(spectra.var(axis=0), 1e-4)
    for c in range(len(classes)):
        rows = spectra[train_labels == c]
        counts.append(len(rows))
        means.append(rows.mean(axis=0))
        # Shrink small-sample estimates towards the training pooled variance.
        var = rows.var(axis=0, ddof=1) if len(rows) > 1 else pooled
        strength = len(rows) / (len(rows) + 10.0)
        variances.append(np.maximum(strength * var + (1 - strength) * pooled, 1e-4))
    stats = {"means": np.asarray(means, np.float32), "variances": np.asarray(variances, np.float32)}
    report = {
        "split_path": str(path), "split_sha256": file_hash(path),
        "shape": list(cube.shape), "train_count": len(train_coords), "test_count": len(test_coords),
        "classes": len(classes), "train_class_counts": counts,
        "raw_image_sha256": array_hash(cube),
        "train_coordinates_sha256": array_hash(train_coords),
        "test_coordinates_sha256": array_hash(test_coords),
        "normalization": {"global_mean": mean, "global_scale": scale, "fit_on": "TR center spectra only"},
        "pca": False,
        "protocol": "original TR/TE; neighborhood inputs may overlap across train/test; not a spatially disjoint or real cross-sensor evaluation",
    }
    if "preparation_json" in obj:
        preparation = json.loads(str(np.asarray(obj["preparation_json"]).squeeze().item()))
        report["dataset_preparation"] = preparation
        report["protocol"] = preparation["protocol"]
    return normalized.astype(np.float32), tr, te, stats, report


def select_evaluation_pixels(te, per_class, seed):
    """Fixed test subset for a pipeline check; never changes training statistics."""
    if per_class < 0:
        raise ValueError("eval_per_class 不能为负数。")
    te = np.asarray(te)
    if per_class == 0:
        selected = te
    else:
        selected = np.zeros_like(te)
        rng = np.random.default_rng(seed)
        for class_id in np.unique(te[te > 0]):
            positions = np.argwhere(te == class_id)
            indices = rng.choice(len(positions), min(per_class, len(positions)), replace=False)
            rows = positions[indices]
            selected[rows[:, 0], rows[:, 1]] = class_id
    return selected, {
        "scope": "all_test_pixels" if per_class == 0 else "fixed_per_class_subset_for_pipeline_check",
        "per_class_limit": per_class, "selection_seed": seed if per_class else None,
        "available_test_count": int(np.count_nonzero(te)),
        "evaluated_test_count": int(np.count_nonzero(selected)),
        "label_mask_sha256": array_hash(selected),
        "interpretation": "subset metrics are not full-scene OA" if per_class else "all supplied TE centers",
    }


class PatchDataset:
    """Crop on demand; no all-patch cache and no global augmentation RNG."""
    def __init__(self, cube, label_mask, patch_size, augment_seed=None):
        import torch
        self.torch = torch
        if patch_size < 1 or patch_size % 2 != 1:
            raise ValueError("patch_size 必须是正奇数。")
        self.coords = np.argwhere(label_mask > 0)
        self.labels = label_mask[label_mask > 0].astype(np.int64) - 1
        self.patch_size = patch_size
        self.margin = patch_size // 2
        self.image = np.pad(cube, ((self.margin, self.margin), (self.margin, self.margin), (0, 0)),
                            mode="reflect")
        self.rng = np.random.default_rng(augment_seed) if augment_seed is not None else None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        row, col = self.coords[index]
        p = self.image[row:row + self.patch_size, col:col + self.patch_size]
        if self.rng is not None:
            p = np.rot90(p, int(self.rng.integers(4)), axes=(0, 1))
        # One shared spatial augmentation precedes both observation views.
        # Length-1 axes can retain negative strides even when NumPy reports
        # C-contiguity; PyTorch rejects those strides. Force a fresh C copy.
        return self.torch.from_numpy(p.transpose(2, 0, 1).copy(order="C")), int(self.labels[index]), index
