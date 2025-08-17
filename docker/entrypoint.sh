#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="${PROJECT_ROOT}/bin:${PATH}"

echo "[entrypoint] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[entrypoint] PYTHONPATH=${PYTHONPATH}"

exec micromamba run -n bulletarm bash