#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/"
    "sample/100BT"
)


def read_manifest(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names or any(Path(name).name != name for name in names):
        raise ValueError(f"Invalid shard manifest: {path}")
    return names


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def download(url: str, destination: Path, expected_size: int) -> None:
    part = destination.with_suffix(destination.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            offset = 0
            part.unlink(missing_ok=True)
        mode = "ab" if offset else "wb"
        with part.open(mode) as output:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    actual_size = part.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Incomplete download for {destination.name}: {actual_size} != {expected_size}"
        )
    os.replace(part, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)
    names = read_manifest(args.manifest)
    sizes = {}
    for name in names:
        url = f"{BASE_URL}/{name}?download=true"
        sizes[name] = remote_size(url)
    required = sum(
        size
        for name, size in sizes.items()
        if not (args.destination / name).exists()
    )
    free = shutil.disk_usage(args.destination).free
    print(
        f"FineWeb subset: {len(names)} shards, "
        f"{sum(sizes.values()) / 2**30:.2f} GiB total, {free / 2**30:.2f} GiB free",
        flush=True,
    )
    if required + 5 * 2**30 > free:
        raise RuntimeError(
            f"Insufficient free space: need {required / 2**30:.2f} GiB plus 5 GiB reserve"
        )

    for index, name in enumerate(names, start=1):
        destination = args.destination / name
        expected = sizes[name]
        if destination.exists() and destination.stat().st_size == expected:
            print(f"[{index}/{len(names)}] verified {name}", flush=True)
            continue
        destination.unlink(missing_ok=True)
        url = f"{BASE_URL}/{name}?download=true"
        for attempt in range(1, 6):
            try:
                print(f"[{index}/{len(names)}] downloading {name}, attempt {attempt}", flush=True)
                download(url, destination, expected)
                break
            except (OSError, urllib.error.URLError, RuntimeError) as exc:
                if attempt == 5:
                    raise
                print(f"retry after error: {exc}", file=sys.stderr, flush=True)
                time.sleep(15 * attempt)

    (args.destination / ".subset_complete").write_text(
        "\n".join(f"{name} {sizes[name]}" for name in names) + "\n"
    )
    print("FineWeb subset is complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
