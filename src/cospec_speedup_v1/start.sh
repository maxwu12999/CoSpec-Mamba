#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
runtime_python=/root/autodl-tmp/venvs/cospec/bin/python
if [[ ! -x "$runtime_python" ]]; then runtime_python=python; fi
exec "$runtime_python" -u launcher.py "$@"
