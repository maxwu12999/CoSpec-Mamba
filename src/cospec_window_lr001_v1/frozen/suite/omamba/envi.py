"""Read ENVI binary arrays in native channel order, without extra dependencies."""
from pathlib import Path
import re

import numpy as np


def read_header(path):
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8-sig")
    if text.splitlines()[0].strip().upper() != "ENVI":
        raise ValueError(f"不是 ENVI 头文件：{path}")
    fields = {}
    pattern = r"(?im)^\s*([^=\r\n{}]+?)\s*=\s*(\{[^}]*\}|[^\r\n]*)"
    for match in re.finditer(pattern, text):
        key = " ".join(match.group(1).lower().split())
        if key in fields:
            raise ValueError(f"HDR 字段重复：{key}")
        fields[key] = match.group(2).strip()
    for key in ("samples", "lines", "bands", "data type", "byte order", "interleave"):
        if key not in fields:
            raise ValueError(f"HDR 缺少 {key}：{path}")
    return fields


def read_envi(header_path, data_path=None):
    """Return an H×W×C memory-mapped view and its parsed header."""
    header_path = Path(header_path).expanduser().resolve()
    data_path = (Path(data_path).expanduser().resolve() if data_path is not None
                 else header_path.with_suffix(""))
    fields = read_header(header_path)
    try:
        width, height, bands = (int(fields[k]) for k in ("samples", "lines", "bands"))
        offset = int(fields.get("header offset", "0"))
        order = int(fields["byte order"])
        type_id = int(fields["data type"])
    except ValueError as error:
        raise ValueError("HDR 尺寸、数据类型或字节序不是整数。") from error
    if min(width, height, bands) < 1 or offset < 0 or order not in (0, 1):
        raise ValueError("HDR 尺寸、偏移或字节序无效。")
    types = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2",
             13: "u4", 14: "i8", 15: "u8"}
    if type_id not in types:
        raise ValueError(f"不支持 ENVI data type={type_id}；需要实数影像或整数标签。")
    dtype = np.dtype(types[type_id]).newbyteorder("<" if order == 0 else ">")
    expected = offset + width * height * bands * dtype.itemsize
    if not data_path.is_file():
        raise ValueError(f"找不到配套二进制文件：{data_path}。HDR 只是说明文件；请指定实际影像或标签。")
    if data_path.stat().st_size != expected:
        raise ValueError(f"二进制文件大小不匹配：{data_path}；需要 {expected} 字节，"
                         f"实际 {data_path.stat().st_size}。请检查配对文件、尺寸、类型及偏移。")
    layout = fields["interleave"].lower()
    shapes = {"bsq": (bands, height, width), "bil": (height, bands, width),
              "bip": (height, width, bands)}
    if layout not in shapes:
        raise ValueError(f"不支持的 ENVI interleave：{layout}")
    raw = np.memmap(data_path, mode="r", dtype=dtype, offset=offset, shape=shapes[layout])
    cube = raw.transpose(1, 2, 0) if layout == "bsq" else (
        raw.transpose(0, 2, 1) if layout == "bil" else raw)
    return cube, fields, data_path


def fixed_class_split(labels, train_per_class, seed):
    """Stratified center-pixel split. Patch neighborhoods are not spatially disjoint."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.isfinite(labels).all():
        raise ValueError("标签必须是有限值的二维数组。")
    if np.any(labels < 0) or np.any(labels != np.round(labels)):
        raise ValueError("标签必须是 0 背景和 1..C 的整数。")
    if train_per_class < 1 or not 0 <= seed < 2 ** 32:
        raise ValueError("每类训练数量必须为正数，划分种子必须在 [0, 2**32)。")
    ids = np.unique(labels[labels > 0])
    if not len(ids) or not np.array_equal(ids, np.arange(1, len(ids) + 1)):
        raise ValueError("地物类别必须连续编号为 1..C，0 为背景。")
    labels = labels.astype(np.int64)
    tr = np.zeros_like(labels)
    rng = np.random.default_rng(seed)
    total_counts = []
    for class_id in ids.astype(int):
        positions = np.argwhere(labels == class_id)
        total_counts.append(len(positions))
        if len(positions) <= train_per_class:
            raise ValueError(f"类别 {class_id} 只有 {len(positions)} 个像素，无法选取 "
                             f"{train_per_class} 个训练像素并保留测试像素。")
        selected = positions[rng.choice(len(positions), train_per_class, replace=False)]
        tr[selected[:, 0], selected[:, 1]] = class_id
    te = labels.copy()
    te[tr > 0] = 0
    return tr, te, {
        "seed": seed, "train_per_class": train_per_class, "classes": len(ids),
        "total_class_counts": total_counts,
        "train_count": int(np.count_nonzero(tr)), "test_count": int(np.count_nonzero(te)),
        "train_class_counts": [train_per_class] * len(ids),
        "test_class_counts": [n - train_per_class for n in total_counts],
        "protocol": "fixed stratified random center-pixel split; all remaining labeled centers "
                    "are TE; neighborhood inputs may overlap; not spatially disjoint",
    }
