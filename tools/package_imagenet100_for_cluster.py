#!/usr/bin/env python3
"""Create a minimal SHA256-checksummed transfer archive for cluster use."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.data_root.resolve()
    output = args.output.resolve()
    required = [root / "train", root / "calibration", root / "test", root / "metadata"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing prepared dataset components: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in required:
            archive.add(path, arcname=Path(root.name) / path.relative_to(root))
    digest = hashlib.sha256()
    with output.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    print(f"archive: {output}")
    print(f"bytes: {output.stat().st_size}")
    print(f"sha256: {digest.hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
