"""Count occurrences of a marker inside every layer of a saved Docker image.

Usage: uv run python scripts/scan_image_layers.py <image> <marker>

Layers are decompressed before searching, because a byte search over a compressed
`docker save` stream finds nothing even when the marker is present. Used to show that
git credentials in the build context reach no layer of the final image
(docs/failures.md); scanning the discarded `source` build stage is the positive control.
"""

from __future__ import annotations

import gzip
import io
import subprocess
import sys
import tarfile

GZIP = b"\x1f\x8b"
ZSTD = b"\x28\xb5\x2f\xfd"


def count(image: str, marker: bytes) -> tuple[int, int]:
    saved = subprocess.run(["docker", "save", image], capture_output=True, check=True).stdout
    blobs = hits = 0
    with tarfile.open(fileobj=io.BytesIO(saved)) as outer:
        for member in outer.getmembers():
            handle = outer.extractfile(member) if member.isfile() else None
            if handle is None:
                continue
            data = handle.read()
            if data.startswith(GZIP):
                data = gzip.decompress(data)
            elif data.startswith(ZSTD):
                raise SystemExit("zstd-compressed layer: install a zstd decoder to scan it")
            blobs += 1
            hits += data.count(marker)
    return blobs, hits


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    blobs, hits = count(argv[1], argv[2].encode())
    print(f"{argv[1]}: {blobs} blobs scanned, {hits} occurrences of {argv[2]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
