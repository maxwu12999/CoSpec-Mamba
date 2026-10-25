"""Explicit relative coordinates or genuine wavelength metadata; keep units distinct."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def relative_index_metadata(channels, dataset=None):
    """Generate coordinates ONCE for the complete supplied cube, before masking.

    These IDs refer to the channels in the supplied MAT, not recovered sensor IDs.
    The coordinates express order; equal index gaps do not imply equal nm gaps.
    """
    if isinstance(channels, bool) or not isinstance(channels, (int, np.integer)) or channels < 2:
        raise ValueError("相对坐标至少需要两个完整输入波段。")
    channels = int(channels)
    coordinates = (np.arange(channels, dtype=np.float64) / (channels - 1)).astype(np.float32)
    metadata = {
        "format": "relative_index_coordinates_v1", "coordinate_mode": "relative_index",
        "path": None, "source": "Original channel order in the complete supplied cube",
        "dataset": dataset,
        "channels": channels, "units": "dimensionless", "physical_wavelengths_used": False,
        "formula": "u_i = i / (C - 1), i = 0,...,C-1",
        "relative_coordinates": coordinates.tolist(), "original_band_ids": list(range(channels)),
        "band_id_reference": "Zero-based channel IDs in the complete supplied cube, not recovered sensor band IDs",
        "missing_band_policy": "Keep the original C coordinates and select observations by a Boolean mask; never renumber",
        "assumption": "Index spacing represents spectral order only; physical wavelength intervals and bandwidths are unknown",
    }
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    metadata["sha256"] = hashlib.sha256(canonical).hexdigest()
    return coordinates, metadata


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata(path, expected_channels):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"波长元数据不存在：{path}。请提供真实波长 JSON、CSV 或 ENVI .hdr。")
    suffix = path.suffix.lower()
    band_ids = None
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if not obj.get("source"):
            raise ValueError("JSON 必须填写 source，说明真实波长的来源。")
        values = obj.get("wavelength_nm")
        if values is None:
            raise ValueError("JSON 必须包含 wavelength_nm 数组，单位为 nm。")
        source = obj["source"]
        band_ids = obj.get("original_band_ids")
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or "wavelength_nm" not in rows[0]:
            raise ValueError("CSV 必须有 wavelength_nm 表头，单位为 nm。")
        values = [float(r["wavelength_nm"]) for r in rows]
        source = f"User supplied CSV: {path.name}"
        if "original_band_id" in rows[0]:
            band_ids = [int(r["original_band_id"]) for r in rows]
    elif suffix == ".hdr":
        content = path.read_text(encoding="utf-8-sig", errors="strict")
        wave = re.search(r"(?im)^\s*wavelength\s*=\s*\{([^}]*)\}", content)
        units = re.search(r"(?im)^\s*wavelength units\s*=\s*([^\r\n]+)", content)
        if not wave or not units:
            raise ValueError("HDR 缺少 wavelength 或 wavelength units，不能从波段数推算波长。")
        values = [float(x) for x in re.split(r"[\s,]+", wave.group(1).strip()) if x]
        unit = units.group(1).strip().strip("{}").strip().lower()
        if unit in ("micrometers", "micrometer", "microns", "um", "µm"):
            values = [x * 1000.0 for x in values]
        elif unit not in ("nanometers", "nanometer", "nm"):
            raise ValueError(f"无法确认 HDR 波长单位：{unit}")
        declared = re.search(r"(?im)^\s*bands\s*=\s*(\d+)", content)
        if declared and int(declared.group(1)) != expected_channels:
            raise ValueError("HDR 的 bands 与当前影像不一致。删除波段后必须同步选择波长。")
        source = f"ENVI header: {path.name}"
    else:
        raise ValueError("元数据只接受 JSON、CSV 或 .hdr。")
    wavelengths = np.asarray(values, dtype=np.float64)
    if wavelengths.shape != (expected_channels,):
        raise ValueError(f"需要 {expected_channels} 个真实波长，实际形状为 {wavelengths.shape}。")
    if not np.all(np.isfinite(wavelengths)) or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("波长必须有限、严格递增，且顺序与影像通道完全一致。")
    if wavelengths.min() < 200 or wavelengths.max() > 30000:
        raise ValueError("波长超出预期 nm 范围；请检查单位。通道序号不是波长。")
    if band_ids is not None:
        if len(band_ids) != expected_channels or len(set(band_ids)) != expected_channels:
            raise ValueError("original_band_ids 必须与波长等长且无重复。")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in band_ids):
            raise ValueError("original_band_ids 必须是非负整数。")
    return wavelengths.astype(np.float32), {
        "path": str(path), "sha256": file_hash(path), "source": source,
        "channels": expected_channels, "units": "nm",
        "wavelength_nm": wavelengths.tolist(), "original_band_ids": band_ids,
        "native_measurement_assumption": "point samples at band centers; no native SRF/bandwidth inference",
    }
