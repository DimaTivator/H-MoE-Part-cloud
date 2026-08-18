#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
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


@dataclass(frozen=True)
class Shard:
    name: str
    size: int | None = None
    sha256: str | None = None


def read_manifest(path: Path) -> list[Shard]:
    shards: list[Shard] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 1:
            shard = Shard(name=fields[0])
        elif len(fields) == 3:
            digest, size_text, name = fields
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"Invalid SHA-256 on line {line_number}: {path}")
            try:
                size = int(size_text)
            except ValueError as exc:
                raise ValueError(f"Invalid size on line {line_number}: {path}") from exc
            if size <= 0:
                raise ValueError(f"Invalid size on line {line_number}: {path}")
            shard = Shard(name=name, size=size, sha256=digest)
        else:
            raise ValueError(
                f"Expected 'filename' or 'sha256 size filename' on line "
                f"{line_number}: {path}"
            )
        if Path(shard.name).name != shard.name or not shard.name.endswith(".parquet"):
            raise ValueError(f"Unsafe shard name on line {line_number}: {path}")
        shards.append(shard)
    if not shards or len({shard.name for shard in shards}) != len(shards):
        raise ValueError(f"Empty or duplicate shard manifest: {path}")
    return shards


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=120) as response:
        return int(response.headers["Content-Length"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_verified(path: Path, shard: Shard) -> bool:
    if not path.is_file() or shard.size is None or path.stat().st_size != shard.size:
        return False
    return shard.sha256 is None or sha256_file(path) == shard.sha256


def download(url: str, destination: Path, shard: Shard) -> None:
    if shard.size is None:
        raise ValueError(f"Missing expected size for {shard.name}")
    part = destination.with_suffix(destination.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > shard.size:
        part.unlink()
        offset = 0
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
    if actual_size != shard.size:
        raise RuntimeError(
            f"Incomplete download for {destination.name}: {actual_size} != {shard.size}"
        )
    if shard.sha256 is not None:
        actual_sha256 = sha256_file(part)
        if actual_sha256 != shard.sha256:
            part.unlink()
            raise RuntimeError(
                f"SHA-256 mismatch for {destination.name}: "
                f"{actual_sha256} != {shard.sha256}"
            )
    os.replace(part, destination)


def download_with_retries(
    index: int,
    total: int,
    shard: Shard,
    destination_dir: Path,
) -> None:
    destination = destination_dir / shard.name
    url = f"{BASE_URL}/{shard.name}?download=true"
    for attempt in range(1, 6):
        try:
            print(
                f"[{index}/{total}] downloading {shard.name}, attempt {attempt}",
                flush=True,
            )
            download(url, destination, shard)
            print(f"[{index}/{total}] verified {shard.name}", flush=True)
            return
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            if attempt == 5:
                raise
            print(f"retry after error: {exc}", file=sys.stderr, flush=True)
            time.sleep(15 * attempt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--completion-marker", default=".subset_complete")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if Path(args.completion_marker).name != args.completion_marker:
        parser.error("--completion-marker must be a filename")

    args.destination.mkdir(parents=True, exist_ok=True)
    raw_shards = read_manifest(args.manifest)
    shards = [
        shard
        if shard.size is not None
        else Shard(
            name=shard.name,
            size=remote_size(f"{BASE_URL}/{shard.name}?download=true"),
            sha256=shard.sha256,
        )
        for shard in raw_shards
    ]

    pending: list[tuple[int, Shard]] = []
    for index, shard in enumerate(shards, start=1):
        destination = args.destination / shard.name
        if is_verified(destination, shard):
            print(f"[{index}/{len(shards)}] verified {shard.name}", flush=True)
        else:
            pending.append((index, shard))

    required = sum(shard.size or 0 for _, shard in pending)
    free = shutil.disk_usage(args.destination).free
    print(
        f"FineWeb snapshot: {len(shards)} shards, "
        f"{sum(shard.size or 0 for shard in shards) / 2**30:.2f} GiB total, "
        f"{required / 2**30:.2f} GiB pending, {free / 2**30:.2f} GiB free",
        flush=True,
    )
    if required + 5 * 2**30 > free:
        raise RuntimeError(
            f"Insufficient free space: need {required / 2**30:.2f} GiB plus 5 GiB reserve"
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_with_retries,
                index,
                len(shards),
                shard,
                args.destination,
            )
            for index, shard in pending
        ]
        for future in as_completed(futures):
            future.result()

    (args.destination / args.completion_marker).write_text(
        "\n".join(
            f"{shard.sha256 or '-'} {shard.size} {shard.name}" for shard in shards
        )
        + "\n"
    )
    print("FineWeb snapshot is complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
