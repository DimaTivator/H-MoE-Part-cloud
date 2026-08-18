#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from transformers import AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.fineweb_streaming_core import (  # noqa: E402
    FineWebEduStream,
    Manifest,
    build_snapshot_split_plan_with_val_blocks,
)


EXPECTED_MANIFEST_FINGERPRINT = (
    "7327154b810ec27cf5ca794aedcc3aea11796b218261ff24b5e2d3d2d283e00b"
)
EXPECTED_SPLIT_PLAN_FINGERPRINT = (
    "550b33876a3810f4bee389f6b897584017bf9f7a6ac4456aa0de9cf043c09455"
)
EXPECTED_VAL_BLOCKS_SHA256 = (
    "d9b18bcef1a4ef61a493dbcf2ebb2afadd8fa2a207111dd004d34761e406448e"
)
EXPECTED_PREVIEW_SHA256 = {
    0: "fef09aaf0c5d7056e2421b34a5e5e9e761932721a6d0b14f65ab2050c4690393",
    1: "f7ae4a8aefabb183355afaa4b7b23193b8e8faf42c634a1bb131e99b3c9819a3",
}
PREVIEW_BLOCKS = 4096
BLOCK_TOKENS = 1025
VAL_BLOCKS = 1024
WORLD_SIZE = 2
ITERATIONS = 75_457
MICROSTEPS_PER_ITERATION = 4
BATCH_SIZE = 16
BLOCKS_PER_RANK = ITERATIONS * MICROSTEPS_PER_ITERATION * BATCH_SIZE
CHECKPOINT_EVERY_BLOCKS = 8192


def load_manifest(path: Path, dataset_root: str | None = None) -> Manifest:
    compressed = base64.b64decode(path.read_text().strip())
    payload = json.loads(gzip.decompress(compressed))
    manifest = Manifest.from_dict(payload)
    if manifest.fingerprint != EXPECTED_MANIFEST_FINGERPRINT:
        raise ValueError(f"Unexpected manifest fingerprint: {manifest.fingerprint}")
    if dataset_root is not None:
        manifest = Manifest(dataset_root=dataset_root, shards=manifest.shards)
    return manifest


def tokenizer_factory():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024
    return tokenizer


def blocks_sha256(blocks: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.sha256()
    for block in blocks:
        digest.update(np.asarray(block, dtype="<u4").tobytes())
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    os.replace(temporary, path)


def build_snapshot(manifest: Manifest):
    snapshot = build_snapshot_split_plan_with_val_blocks(
        manifest,
        tokenizer_factory,
        block_tokens=BLOCK_TOKENS,
        val_sequences=VAL_BLOCKS,
        split_seed=2357,
        shuffle_seed=1337,
    )
    if snapshot.plan.fingerprint != EXPECTED_SPLIT_PLAN_FINGERPRINT:
        raise ValueError(f"Unexpected split fingerprint: {snapshot.plan.fingerprint}")
    validation_sha256 = blocks_sha256(snapshot.val_blocks)
    if validation_sha256 != EXPECTED_VAL_BLOCKS_SHA256:
        raise ValueError(f"Unexpected validation token hash: {validation_sha256}")
    return snapshot


def make_stream(manifest: Manifest, split_plan, rank: int) -> FineWebEduStream:
    return FineWebEduStream(
        manifest,
        tokenizer_factory,
        block_tokens=BLOCK_TOKENS,
        split_plan=split_plan,
        split="train",
        rank=rank,
        world_size=WORLD_SIZE,
        worker_id=0,
        num_data_workers=1,
        num_token_workers=8,
        doc_batch_size=64,
        prefetch_batches=4,
    )


def preview_hash(manifest: Manifest, split_plan, rank: int, blocks: int) -> str:
    digest = hashlib.sha256()
    with make_stream(manifest, split_plan, rank) as stream:
        for _ in range(blocks):
            block = next(stream)
            digest.update(np.asarray(block, dtype="<u2").tobytes())
    return digest.hexdigest()


def write_validation(destination: Path, validation_blocks) -> dict[str, Any]:
    path = destination / "validation.uint16"
    payload = np.asarray(validation_blocks, dtype="<u2").tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    metadata = {
        "file": path.name,
        "blocks": VAL_BLOCKS,
        "bytes": len(payload),
        "sha256": digest,
        "token_sha256_uint32": EXPECTED_VAL_BLOCKS_SHA256,
    }
    write_json_atomic(destination / "validation.json", metadata)
    return metadata


def maybe_finalize(destination: Path) -> None:
    metadata_paths = [destination / f"train_rank{rank}.json" for rank in range(WORLD_SIZE)]
    validation_path = destination / "validation.json"
    if not validation_path.is_file() or not all(path.is_file() for path in metadata_paths):
        return
    ranks = [json.loads(path.read_text()) for path in metadata_paths]
    validation = json.loads(validation_path.read_text())
    metadata = {
        "format": "packed_fineweb_h200_v1",
        "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
        "split_plan_fingerprint": EXPECTED_SPLIT_PLAN_FINGERPRINT,
        "validation_blocks_sha256": EXPECTED_VAL_BLOCKS_SHA256,
        "world_size": WORLD_SIZE,
        "sequence_length": BLOCK_TOKENS - 1,
        "block_tokens": BLOCK_TOKENS,
        "batch_size": BATCH_SIZE,
        "microsteps_per_iteration": MICROSTEPS_PER_ITERATION,
        "iterations": ITERATIONS,
        "blocks_per_rank": BLOCKS_PER_RANK,
        "ranks": ranks,
        "validation": validation,
    }
    write_json_atomic(destination / "packed_metadata.json", metadata)
    print("PACKED_SNAPSHOT=complete", flush=True)


def prepare_rank(destination: Path, manifest: Manifest, split_plan, rank: int) -> None:
    final_path = destination / f"train_rank{rank}.uint16"
    rank_metadata_path = destination / f"train_rank{rank}.json"
    if final_path.is_file() and rank_metadata_path.is_file():
        print(f"RANK={rank} already complete", flush=True)
        maybe_finalize(destination)
        return

    part_path = final_path.with_suffix(final_path.suffix + ".part")
    state_path = destination / f"train_rank{rank}.state.json"
    stream = make_stream(manifest, split_plan, rank)
    blocks_written = 0
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        blocks_written = int(state["blocks_written"])
        expected_bytes = blocks_written * BLOCK_TOKENS * 2
        if not part_path.is_file() or part_path.stat().st_size < expected_bytes:
            raise RuntimeError(f"Rank {rank} partial data is inconsistent with resume state")
        with part_path.open("r+b") as output:
            output.truncate(expected_bytes)
        stream.load_state_dict(state["stream_state"])
        print(f"RANK={rank} resuming at block {blocks_written}", flush=True)
    else:
        part_path.write_bytes(b"")

    digest = hashlib.sha256()
    if blocks_written:
        with part_path.open("rb") as source:
            while chunk := source.read(16 * 1024 * 1024):
                digest.update(chunk)

    try:
        with part_path.open("ab") as output:
            while blocks_written < BLOCKS_PER_RANK:
                count = min(CHECKPOINT_EVERY_BLOCKS, BLOCKS_PER_RANK - blocks_written)
                blocks = [next(stream) for _ in range(count)]
                if max(max(block) for block in blocks) >= 65536:
                    raise ValueError("GPT-2 token id does not fit uint16")
                array = np.asarray(blocks, dtype="<u2")
                payload = array.tobytes()
                if blocks_written == 0:
                    preview_bytes = PREVIEW_BLOCKS * BLOCK_TOKENS * 2
                    preview_sha256 = hashlib.sha256(payload[:preview_bytes]).hexdigest()
                    if preview_sha256 != EXPECTED_PREVIEW_SHA256[rank]:
                        raise ValueError(
                            f"Rank {rank} preview hash does not match H200: {preview_sha256}"
                        )
                output.write(payload)
                output.flush()
                digest.update(payload)
                blocks_written += count
                state = {
                    "blocks_written": blocks_written,
                    "stream_state": stream.state_dict(),
                }
                write_json_atomic(state_path, state)
                print(
                    f"RANK={rank} blocks={blocks_written}/{BLOCKS_PER_RANK} "
                    f"bytes={part_path.stat().st_size}",
                    flush=True,
                )
    finally:
        stream.close()

    expected_bytes = BLOCKS_PER_RANK * BLOCK_TOKENS * 2
    if part_path.stat().st_size != expected_bytes:
        raise RuntimeError(f"Unexpected packed size for rank {rank}")
    os.replace(part_path, final_path)
    metadata = {
        "rank": rank,
        "file": final_path.name,
        "blocks": BLOCKS_PER_RANK,
        "bytes": expected_bytes,
        "sha256": digest.hexdigest(),
        "preview_blocks": PREVIEW_BLOCKS,
        "preview_sha256": EXPECTED_PREVIEW_SHA256[rank],
    }
    write_json_atomic(rank_metadata_path, metadata)
    state_path.unlink(missing_ok=True)
    print(f"RANK={rank} sha256={metadata['sha256']}", flush=True)
    maybe_finalize(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--rank", type=int, choices=range(WORLD_SIZE))
    parser.add_argument("--preview-blocks", type=int, default=0)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, dataset_root=args.dataset_root)
    snapshot = build_snapshot(manifest)
    print(f"manifest_fingerprint={manifest.fingerprint}", flush=True)
    print(f"split_plan_fingerprint={snapshot.plan.fingerprint}", flush=True)
    print(f"validation_blocks_sha256={EXPECTED_VAL_BLOCKS_SHA256}", flush=True)

    if args.preview_blocks:
        if args.rank is None or args.preview_blocks <= 0:
            parser.error("--preview-blocks requires --rank and a positive value")
        digest = preview_hash(manifest, snapshot.plan, args.rank, args.preview_blocks)
        if args.preview_blocks == PREVIEW_BLOCKS:
            expected = EXPECTED_PREVIEW_SHA256[args.rank]
            if digest != expected:
                raise ValueError(f"Rank {args.rank} preview hash does not match H200")
        print(
            f"RANK={args.rank} PREVIEW_BLOCKS={args.preview_blocks} SHA256={digest}",
            flush=True,
        )
        return 0

    if args.rank is None or args.destination is None:
        parser.error("full preparation requires --rank and --destination")
    args.destination.mkdir(parents=True, exist_ok=True)
    if args.rank == 0:
        write_validation(args.destination, snapshot.val_blocks)
    prepare_rank(args.destination, manifest, snapshot.plan, args.rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
