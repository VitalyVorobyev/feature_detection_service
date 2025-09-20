#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGETS=(
  fds.py
  fds_cli.py
  features.py
  test
)

pylint "${TARGETS[@]}"
