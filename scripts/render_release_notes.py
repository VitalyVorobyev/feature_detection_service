#!/usr/bin/env python3
"""Render release notes using the changelog entry for a given version."""
from __future__ import annotations

import argparse
from pathlib import Path


REPO = "image_store_service"
IMAGE_NAME = "ghcr.io/{owner}/image_store_service"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render release notes")
    parser.add_argument("--version", required=True, help="Release version, e.g. v1.0.0")
    parser.add_argument(
        "--image-digest",
        required=True,
        help="OCI image digest reference (e.g. sha256:abcdef)",
    )
    parser.add_argument(
        "--owner",
        required=True,
        help="GitHub repository owner (used for GHCR references)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/release-notes.md"),
        help="Path to write rendered release notes",
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=Path("CHANGELOG.md"),
        help="Changelog file to parse",
    )
    return parser.parse_args()


def extract_changelog_section(changelog: Path, version: str) -> str:
    lines = changelog.read_text(encoding="utf-8").splitlines()
    header = f"## [{version}]"
    collecting = False
    buffer: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if line.startswith(header):
                collecting = True
                buffer.append(line)
                continue
            elif collecting:
                break
        elif collecting:
            buffer.append(line)
    if not buffer:
        raise ValueError(f"Version {version} not found in {changelog}")
    # Trim trailing empty lines
    while buffer and buffer[-1] == "":
        buffer.pop()
    return "\n".join(buffer)


def render_notes(version: str, owner: str, digest: str, changelog_section: str) -> str:
    image_ref = IMAGE_NAME.format(owner=owner)
    lines = [changelog_section, "", "---", "", "## Deployment", "",]
    lines.extend(
        [
            f"- Image: `{image_ref}:{version}`",
            f"- Digest: `{image_ref}@{digest}`",
            "- Cosign: Keyless signature via GitHub OIDC (attached attestation)",
            "- SBOM: `sbom.spdx.json` (SPDX JSON)",
            "- Provenance: `provenance.intoto.jsonl` (SLSA provenance attestation)",
            "- Example config: `example.env`",
            "- Compose snippet: `compose.service.yaml`",
            "- Debug bundle: `dist.tar.gz`",
        ]
    )
    lines.extend(
        [
            "",
            "## Operations",
            "",
            "- Health endpoint: `GET /health`",
            "- Storage: File-system blobs at `ISS_DATA` (no DB migrations required)",
            "- Backward compatibility: Compatible with previous `ISS_DATA` contents",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    section = extract_changelog_section(args.changelog, args.version)
    notes = render_notes(
        version=args.version,
        owner=args.owner,
        digest=args.image_digest,
        changelog_section=section,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
