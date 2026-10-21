#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
runtime_python=python
if [[ -z "${VIRTUAL_ENV:-}" && -x /root/autodl-tmp/venvs/cospec/bin/python ]]; then
  runtime_python=/root/autodl-tmp/venvs/cospec/bin/python
fi
exec "$runtime_python" -u control.py start "$@"
