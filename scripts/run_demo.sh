#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_DIR}"
if ! "${PYTHON_BIN}" -c 'import numpy, pandas, torch' >/dev/null 2>&1; then
  echo "Demo dependencies are missing for: ${PYTHON_BIN}" >&2
  echo "Activate the project environment and run: pip install -e '.[detection]'" >&2
  echo "Or set PYTHON_BIN=/path/to/python before this command." >&2
  exit 1
fi
exec "${PYTHON_BIN}" scripts/run_demo.py "$@"
