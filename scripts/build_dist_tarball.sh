#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="dist"
BUNDLE_NAME="dist.tar.gz"

mkdir -p "$DIST_DIR"

# Build a lightweight bundle for debugging or offline inspection.
tar -czf "$DIST_DIR/$BUNDLE_NAME" \
    README.md \
    requirements.txt \
    fds.py \
    example.env

echo "Created $DIST_DIR/$BUNDLE_NAME"
