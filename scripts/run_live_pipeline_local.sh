#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/joe/Desktop/McGill/projects/FIFAproject2026"
LOCK_DIR="/tmp/cupcast_live_pipeline.lock"

echo ""
echo "[$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')] Starting CupCast live pipeline"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another live pipeline run is already active; skipping this interval."
  exit 0
fi
trap 'rmdir "$LOCK_DIR"' EXIT

cd "$PROJECT_DIR"

export PYTHONPATH=.
export HTTPS_PROXY=http://127.0.0.1:12334
export HTTP_PROXY=http://127.0.0.1:12334
export CUPCAST_API_PACKAGE_LIVE_DIR="/Users/joe/Desktop/FIFAproject2026/cupcast_api_package/output/live"

PYTHON_BIN="/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="/usr/bin/python3"
fi

"$PYTHON_BIN" \
  src/live/update_live_pipeline.py \
  --skip-cleanup \
  --live-sim-n-sims 10000

echo "[$(/bin/date -u '+%Y-%m-%dT%H:%M:%SZ')] Finished CupCast live pipeline"
