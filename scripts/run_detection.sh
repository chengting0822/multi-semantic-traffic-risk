#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 INPUT_VIDEO YOLO_MODEL [OUTPUT_CSV]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_VIDEO="$1"
YOLO_MODEL="$2"
OUTPUT_CSV="${3:-${PROJECT_DIR}/outputs/tracks.csv}"

cd "${PROJECT_DIR}"
PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  exec "${PYTHON_BIN}" -m traffic_risk.cli detect -- \
    "${INPUT_VIDEO}" \
    --yolo-model "${YOLO_MODEL}" \
    --output "${OUTPUT_CSV}"
