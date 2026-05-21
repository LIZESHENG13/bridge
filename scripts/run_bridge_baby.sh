#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../src"
python main.py --model BRIDGE --dataset baby --gpu_id 0 --seed 2020

