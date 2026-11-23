#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${ADAS_PYTHON:-/home/seeed/rockchip-yolo11-quant-benchmark/.venv/bin/python}"
export PYTHONPATH="$PWD/.deps${ADAS_REUSE_DEPS:+:$ADAS_REUSE_DEPS}${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" server.py --host "${ADAS_HOST:-0.0.0.0}" --port "${ADAS_PORT:-8092}"
