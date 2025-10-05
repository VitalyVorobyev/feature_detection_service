#!/usr/bin/env python3
"""Generate the OpenAPI schema for the Image Store Service."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi.openapi.utils import get_openapi


def _root_dir() -> Path:
    # Ensure the project root (which holds `iss.py`) is importable.
    return Path(__file__).resolve().parent.parent


sys.path.insert(0, str(_root_dir()))

from fds import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OpenAPI schema JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/openapi.json"),
        help="Where to write the OpenAPI document (default: dist/openapi.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schema = get_openapi(
        title=app.title,
        version="0.1.0",
        routes=app.routes,
        description=app.description,
    )
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(schema, fp, indent=2, sort_keys=True)
        fp.write("\n")


if __name__ == "__main__":
    main()
