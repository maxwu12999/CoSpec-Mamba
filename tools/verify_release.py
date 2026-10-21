#!/usr/bin/env python3
"""Verify the published file manifest and the restored scientific source hashes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "cospec_full_lr001_v1": "40a0f0ccaa744c3d0c748caacda4a53f0718d6c3a4a0429a26d70ff9e1c2f841",
    "cospec_window_lr001_v1": "08720cb5be7a19b6a315f6761e73537107d46012d4906b7b42c8e9f5805866ef",
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-upstreams", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"release file differs: {name}")
    print(f"Verified {len(manifest['files'])} published files.")
    if not args.with_upstreams:
        return
    for name, expected in EXPECTED.items():
        core = ROOT / "src" / name
        for relative, wanted in json.loads((core / "FROZEN_SHA256.json").read_text()).items():
            if file_hash(core / "frozen" / relative) != wanted:
                raise ValueError(f"frozen file differs: {name}/{relative}")
        identities = ["FROZEN_SHA256.json", "ORIGINAL_PLANS.json", "REFERENCE_DATA.json"]
        if "window" in name:
            identities.append("UPSTREAM_SCIENCE.json")
        paths = list(core.rglob("*.py")) + [core / item for item in identities]
        mapping = {str(path.relative_to(core)): file_hash(path) for path in sorted(paths)}
        encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()
        if hashlib.sha256(encoded).hexdigest() != expected:
            raise ValueError(f"scientific source identity differs: {name}")
        print("Verified archived scientific identity:", name)


if __name__ == "__main__":
    main()

