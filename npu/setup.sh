#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NPU_PYTHON="${NPU_PYTHON:-python3}"
if [ -z "${CANN_ROOT:-}" ]; then
    for candidate in /usr/local/Ascend/cann-9.1.0 /usr/local/Ascend/ascend-toolkit/latest; do
        if [ -f "$candidate/set_env.sh" ]; then CANN_ROOT="$candidate"; break; fi
    done
fi
if [ ! -f "${CANN_ROOT:-}/set_env.sh" ]; then
    echo "CANN not found. Set CANN_ROOT to the installed CANN toolkit root." >&2
    exit 2
fi
source "$CANN_ROOT/set_env.sh"
export CANN_ROOT
unset PYTHONPATH PYTHONHOME LD_PRELOAD
export PATH="$("$NPU_PYTHON" -c 'import sys;from pathlib import Path;print(Path(sys.executable).parent)'):$PATH"
exec "$NPU_PYTHON" "$REPO_ROOT/npu/install.py" "$@"
