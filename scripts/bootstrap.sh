#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    uv venv .venv --python 3.12
fi

uv pip install --python .venv/bin/python -r requirements-dev.txt
