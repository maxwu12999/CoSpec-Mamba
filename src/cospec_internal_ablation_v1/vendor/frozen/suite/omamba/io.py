"""Atomic writes, complete resumable state, bounded result ZIP."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import zipfile

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def model_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def seed_all(seed, warn_only=True):
    """Seed every stream and pin the deterministic kernel paths.

    cudnn.deterministic alone is NOT enough: the selective-scan backward and
    other atomicAdd kernels are only made reproducible by torch's global
    deterministic-algorithms switch. Without it a resumed run can differ from
    the uninterrupted one in the last bits, which is exactly the failure this
    suite has to rule out.
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pin_deterministic(warn_only=warn_only)


def pin_deterministic(warn_only=True):
    """Force the reproducible kernel regime without touching seed state.

    Kept separate from :func:`seed_all` so the resume branch of ``train`` can
    re-assert it: resuming does not re-seed, but it must still run under the
    same deterministic switches as the original process.
    """
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def capture_rng():
    """Snapshot every random stream that training depends on.

    ``cuda`` is stored as a list of CPU tensors so the payload stays loadable
    on a machine without the original GPU, and so ``torch.save`` never trips
    over device-bound RNG state.
    """
    import torch
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
        "cuda": [x.cpu() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    """Restore every stream captured by :func:`capture_rng`.

    Raises when a GPU run resumes a CPU-only checkpoint (or vice versa); a
    silent mismatch would change the dropout stream and break bit-exactness.
    """
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    saved = state.get("cuda") or []
    available = torch.cuda.is_available()
    if saved and not available:
        raise RuntimeError("检查点保存了CUDA随机状态，但当前没有可用CUDA设备；拒绝以不同设备续跑。")
    if saved:
        if len(saved) != torch.cuda.device_count():
            raise RuntimeError(f"检查点的CUDA设备数({len(saved)})与当前({torch.cuda.device_count()})不一致；拒绝续跑。")
        torch.cuda.set_rng_state_all([x.cpu() for x in saved])


def save_checkpoint(path, model, optimizer, epoch, recipe, sampler, dataset, loader_generator, extra):
    import torch
    payload = {
        "format": "structure_mamba_v2", "epoch": epoch, "recipe": recipe,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng(),
        "view_rng": sampler.rng.bit_generator.state,
        "augmentation_rng": dataset.rng.bit_generator.state if dataset.rng else None,
        "loader_rng": loader_generator.get_state(), "extra": extra,
    }
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def package_results(directory, archive_predictions=False):
    directory = Path(directory).resolve()
    archive = directory.with_name(directory.name + "_results.zip")
    temp = archive.with_name(archive.name + ".tmp")
    excluded = []
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in sorted(directory.rglob("*")):
            if not p.is_file():
                continue
            if (p.suffix.lower() in (".pt", ".pth", ".ckpt", ".tmp", ".pyc", ".zip") or
                    "__pycache__" in p.parts or
                    (p.suffix.lower() == ".npz" and not archive_predictions)):
                excluded.append(str(p.relative_to(directory)))
                continue
            z.write(p, str(Path(directory.name) / p.relative_to(directory)))
        z.writestr(directory.name + "/archive_info.json", json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "excluded": excluded, "weights_kept_locally": True,
            "predictions_kept_locally": True,
            "predictions_included": bool(archive_predictions),
        }, ensure_ascii=False, indent=2))
    temp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("结果 ZIP 的 CRC 检查失败。")
    print(f"结果 ZIP：{archive}", flush=True)
    return archive
