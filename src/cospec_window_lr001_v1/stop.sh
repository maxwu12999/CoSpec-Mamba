#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
runtime_python="${COSPEC_PYTHON:-python}"
if [[ -z "${COSPEC_PYTHON:-}" && -x /root/autodl-tmp/venvs/cospec/bin/python ]]; then
  runtime_python=/root/autodl-tmp/venvs/cospec/bin/python
fi
exec "$runtime_python" -u control.py stop "$@"
