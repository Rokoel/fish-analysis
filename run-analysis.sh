#!/usr/bin/env bash
set -euo pipefail
analysis_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$analysis_root/.venv/bin/python" ]]; then
    echo "Create the .venv environment as described in docs/fish-analysis.md." >&2
    exit 1
fi
cd -- "$analysis_root"
exec "$analysis_root/.venv/bin/python" -m fish_viewer.analysis "$@"
