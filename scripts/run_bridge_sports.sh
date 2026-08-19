#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../src"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-2020}"
exec "${PYTHON_BIN}" main.py --model BRIDGE --dataset sports --gpu_id "${GPU_ID}" --seed "${SEED}"
