"""Load the unchanged research implementation without external model downloads."""
from pathlib import Path
import sys

import numpy as np

SUITE = Path(__file__).resolve().parents[1] / "src/cospec_full_lr001_v1/frozen/suite"
if str(SUITE) not in sys.path:
    sys.path.insert(0, str(SUITE))
from suite_models import SPECS, build_model as _build_model


def build_model(wavelengths, class_means, class_variances, *, variant="D",
                patch_size=15, seed=1, backend="reference", dim=64, depth=4,
                basis_size=64):
    """Construct an archived model; input is [B,C,H,W] plus a boolean [B,C] mask.

    Pass measured wavelengths when available, or explicitly defined increasing
    relative band coordinates. Statistics must be fitted to training data only.
    D is the main operator model; B is the mask-aware linear encoding control.
    ``reference`` runs on CPU for checks; formal experiments use ``cuda``.
    """
    wave = np.asarray(wavelengths, dtype=np.float32)
    means = np.asarray(class_means, dtype=np.float32)
    variances = np.asarray(class_variances, dtype=np.float32)
    if wave.ndim != 1 or len(wave) < 2 or not np.isfinite(wave).all() or not np.all(np.diff(wave) > 0):
        raise ValueError("wavelengths must be a finite, strictly increasing vector")
    if means.ndim != 2 or means.shape[1] != len(wave) or variances.shape != means.shape:
        raise ValueError("class statistics must have shape [classes, channels]")
    if not np.isfinite(means).all() or not np.isfinite(variances).all() or np.any(variances <= 0):
        raise ValueError("class statistics must be finite, with positive variances")
    if patch_size < 1 or patch_size % 2 != 1:
        raise ValueError("patch_size must be a positive odd integer")
    if variant not in SPECS:
        raise ValueError(f"unknown variant {variant!r}; choose from {tuple(SPECS)}")
    if backend not in ("reference", "cuda"):
        raise ValueError("backend must be reference or cuda")
    return _build_model(variant, wave, {"means": means, "variances": variances}, seed,
                       dict(patch=patch_size, dim=dim, depth=depth,
                            basis=basis_size, backend=backend))

