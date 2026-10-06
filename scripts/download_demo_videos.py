#!/usr/bin/env python3
"""Download the four original demo inputs from the GitHub Release."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


BASE_URL = (
    "https://github.com/chengting0822/multi-semantic-traffic-risk/"
    "releases/download/demo-v1"
)
FILES = {
    "14.mp4": "2f166bfec80d94265915e24b27c8b05c7539f828a4a060f181e338c52cbbb802",
    "76.mp4": "23f282aa67142cde900c50fbda09c6e13fb9542551da111f9fddd0f58d1ac0cd",
    "96.mp4": "dc3ce7221f1ac6503a848cf5848140944b7164ed4ed054e76f84da40b0b36549",
    "115.mp4": "ed55b362396c09b3de5db1917fe3f70ffcd9903e4de8b838660a07b5d111e558",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo_videos"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for filename, expected in FILES.items():
        target = args.output / filename
        if target.exists() and not args.force and sha256(target) == expected:
            print(f"ok       {target}")
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        print(f"download {filename}")
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", temporary)
        actual = sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {filename}: {actual}")
        temporary.replace(target)
        print(f"written  {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
