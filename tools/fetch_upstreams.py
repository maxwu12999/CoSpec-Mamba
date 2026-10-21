#!/usr/bin/env python3
"""Restore pinned third-party files; verify bytes before writing any destination."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CORES = (ROOT / "src/cospec_full_lr001_v1", ROOT / "src/cospec_window_lr001_v1")
PREFIX = "# Network class extracted verbatim from the notebook linked by the original HybridSN repository.\n"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def derive_hybridsn(data):
    notebook = json.loads(data)
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "class HybridSN(" not in source:
            continue
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "HybridSN")
        return (PREFIX + "\n".join(source.splitlines()[cls.lineno - 1:cls.end_lineno]) + "\n").encode()
    raise ValueError("HybridSN class not found in pinned notebook")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify local sources without network access")
    args = parser.parse_args()
    entries = json.loads((ROOT / "tools/upstream_sources.json").read_text())
    downloaded = {}
    for entry in entries:
        paths = [core / "frozen" / entry["target"] for core in CORES]
        blob = next((p.read_bytes() for p in paths if p.is_file() and sha(p.read_bytes()) == entry["sha256"]), None)
        if blob is None:
            if args.check:
                raise FileNotFoundError(f"missing or changed: {entry['target']}; run without --check to fetch")
            url = (f"https://raw.githubusercontent.com/{entry['repository']}/"
                   f"{entry['commit']}/{entry['upstream_path']}")
            request = urllib.request.Request(url, headers={"User-Agent": "CoSpec-Mamba-source-restorer"})
            with urllib.request.urlopen(request, timeout=60) as response:
                blob = response.read()
            if sha(blob) != entry["sha256"]:
                raise ValueError(f"checksum mismatch; no file written: {entry['target']}")
        for path in paths:
            if args.check:
                if not path.is_file() or sha(path.read_bytes()) != entry["sha256"]:
                    raise ValueError(f"missing or changed: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
        downloaded[entry["target"]] = blob
        print("verified", entry["target"], flush=True)
    derived = derive_hybridsn(downloaded["external/vendor/hybridsn/HybridSN.ipynb"])
    for core in CORES:
        relative = "external/vendor/hybridsn/HybridSN.py"
        expected = json.loads((core / "FROZEN_SHA256.json").read_text())[relative]
        if sha(derived) != expected:
            raise ValueError("HybridSN class extraction differs from the research archive")
        path = core / "frozen" / relative
        if args.check:
            if not path.is_file() or sha(path.read_bytes()) != expected:
                raise ValueError(f"missing or changed: {path}")
        else:
            path.write_bytes(derived)
    print("All pinned third-party files match the research archive.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Source restoration failed: {error}", file=sys.stderr)
        raise SystemExit(1)

