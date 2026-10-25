"""Only complete observations and random band deletion; separate RNG streams."""
import hashlib

import numpy as np


def mask_hash(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def make_mask(wavelengths, keep_fraction, rng):
    channels = len(wavelengths)
    if channels < 2 or not 0 < keep_fraction <= 1:
        raise ValueError("至少需要两个波段，keep_fraction 必须在 (0,1]。")
    keep = max(1, min(channels, int(round(channels * keep_fraction))))
    mask = np.zeros(channels, dtype=bool)
    mask[rng.choice(channels, keep, replace=False)] = True
    return mask


def evaluation_cases(wavelengths, seed=20261007, repeats=1, coordinate_metadata=None):
    if repeats < 1:
        raise ValueError("repeats 必须是正整数。")
    rng = np.random.default_rng(seed)
    full = [True] * len(wavelengths)
    cases = [{"name": "full", "kind": "full", "keep_fraction": 1.0,
              "mask": full, "sha256": mask_hash(full)}]
    used = {mask_hash(full)}
    for fraction in (0.8, 0.6, 0.4):
        for rep in range(repeats):
            for _ in range(1000):
                mask = make_mask(wavelengths, fraction, rng)
                key = mask_hash(mask)
                if key not in used:
                    break
            else:
                raise ValueError("独立随机评估掩码不足；减少 repeats 或增加波段数。")
            used.add(key)
            cases.append({"name": f"random_keep{int(mask.sum())}_r{rep}",
                          "kind": "random", "keep_fraction": float(mask.mean()),
                          "mask": mask.tolist(), "sha256": key})
    if coordinate_metadata is not None:
        coordinates = np.asarray(coordinate_metadata["relative_coordinates"], dtype=np.float32)
        if not np.array_equal(coordinates, np.asarray(wavelengths, dtype=np.float32)):
            raise ValueError("评估坐标与完整输入坐标不一致。")
        for case in cases:
            selected = np.flatnonzero(case["mask"])
            case.update(coordinate_mode=coordinate_metadata["coordinate_mode"],
                        coordinate_units=coordinate_metadata["units"],
                        coordinate_sha256=coordinate_metadata["sha256"],
                        original_band_ids=selected.tolist(),
                        observed_relative_coordinates=coordinates[selected].tolist())
    return cases


class ObservationSampler:
    def __init__(self, wavelengths, seed, forbidden_masks=(), min_keep=0.6, max_keep=0.9):
        if not 0 < min_keep <= max_keep < 1:
            raise ValueError("训练保留比例必须满足 0 < min_keep <= max_keep < 1。")
        self.wavelengths = np.asarray(wavelengths)
        self.rng = np.random.default_rng(seed)
        self.forbidden = {mask_hash(x) for x in forbidden_masks}
        self.min_keep, self.max_keep = min_keep, max_keep

    def draw(self, batch_size, device):
        import torch
        masks = []
        for _ in range(batch_size):
            for _ in range(1000):
                fraction = float(self.rng.uniform(self.min_keep, self.max_keep))
                mask = make_mask(self.wavelengths, fraction, self.rng)
                if mask.all():
                    mask[self.rng.integers(len(mask))] = False
                if mask_hash(mask) not in self.forbidden:
                    masks.append(mask)
                    break
            else:
                raise RuntimeError("无法生成与固定评估方案不同的随机训练掩码。")
        return torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device)
