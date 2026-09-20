#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 C4O_FEATURES.csv TRAJECTORY_FEATURES.csv TIMESTAMP_DIR OUTPUT_DIR [DEVICE]" >&2
  exit 2
fi

C4O_FEATURES=$1
TRAJECTORY_FEATURES=$2
TIMESTAMP_DIR=$3
OUTPUT_DIR=$4
DEVICE=${5:-auto}

traffic-risk run-risk \
  "$C4O_FEATURES" \
  "$TRAJECTORY_FEATURES" \
  "$TIMESTAMP_DIR" \
  "$OUTPUT_DIR" \
  --device "$DEVICE"
