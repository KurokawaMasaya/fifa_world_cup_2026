#!/bin/zsh
set -e

cd /Users/joe/Desktop/McGill/projects/FIFAproject2026

export PYTHONPATH=.
export HTTPS_PROXY=http://127.0.0.1:12334
export HTTP_PROXY=http://127.0.0.1:12334

/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  src/live/update_live_pipeline.py \
  --skip-cleanup \
  --live-sim-n-sims 10000
