#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from transformers import AutoTokenizer

from src.data.fineweb import DEFAULT_FINEWEB_SPLIT_SEED
from src.data.fineweb_streaming_core import (
    RowGroupRef,
    build_manifest,
    build_snapshot_split_plan_with_val_blocks,
)


EXPECTED_MANIFEST_FINGERPRINT = (
    "7327154b810ec27cf5ca794aedcc3aea11796b218261ff24b5e2d3d2d283e00b"
)
EXPECTED_SPLIT_PLAN_FINGERPRINT = (
    "550b33876a3810f4bee389f6b897584017bf9f7a6ac4456aa0de9cf043c09455"
)
EXPECTED_SHARDS = 140
EXPECTED_ROWS = 97_270_686
EXPECTED_ROW_GROUPS = 97_277
EXPECTED_VAL_ROW_GROUPS = (
    RowGroupRef("008_00004.parquet", 349),
    RowGroupRef("006_00006.parquet", 400),
)
EXPECTED_VAL_BLOCKS_SHA256 = (
    "d9b18bcef1a4ef61a493dbcf2ebb2afadd8fa2a207111dd004d34761e406448e"
)


def blocks_fingerprint(blocks: tuple[tuple[int, ...], ...]) -> str:
    digest = hashlib.sha256()
    for block in blocks:
        for token in block:
            digest.update(int(token).to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    if args.world_size != 2:
        parser.error("H200 data parity requires --world-size 2")

    manifest = build_manifest(args.dataset_dir)
    tokenizer_factory = lambda: AutoTokenizer.from_pretrained("gpt2")
    snapshot = build_snapshot_split_plan_with_val_blocks(
        manifest,
        tokenizer_factory,
        block_tokens=1025,
        val_sequences=1024,
        split_seed=DEFAULT_FINEWEB_SPLIT_SEED,
        shuffle_seed=1337,
    )
    plan = snapshot.plan

    rows = sum(shard.num_rows for shard in manifest.shards)
    assert len(manifest.shards) == EXPECTED_SHARDS, len(manifest.shards)
    assert rows == EXPECTED_ROWS, rows
    assert manifest.total_row_groups == EXPECTED_ROW_GROUPS, manifest.total_row_groups
    assert manifest.fingerprint == EXPECTED_MANIFEST_FINGERPRINT, manifest.fingerprint
    assert plan.fingerprint == EXPECTED_SPLIT_PLAN_FINGERPRINT, plan.fingerprint
    assert plan.val_row_groups == EXPECTED_VAL_ROW_GROUPS, plan.val_row_groups
    assert len(snapshot.val_blocks) == 1024, len(snapshot.val_blocks)
    val_blocks_sha256 = blocks_fingerprint(snapshot.val_blocks)
    assert val_blocks_sha256 == EXPECTED_VAL_BLOCKS_SHA256, val_blocks_sha256

    print(f"manifest_fingerprint={manifest.fingerprint}")
    print(f"split_plan_fingerprint={plan.fingerprint}")
    print(f"shards={len(manifest.shards)} rows={rows} row_groups={manifest.total_row_groups}")
    print(
        "validation_row_groups="
        + ",".join(
            f"{row_group.relative_path}#{row_group.row_group_index}"
            for row_group in plan.val_row_groups
        )
    )
    print(f"validation_blocks_sha256={val_blocks_sha256}")
    for rank in range(args.world_size):
        assigned = plan.train_row_groups[rank :: args.world_size]
        preview = ",".join(
            f"{row_group.relative_path}#{row_group.row_group_index}"
            for row_group in assigned[:5]
        )
        print(f"rank={rank} first_train_row_groups={preview}")
    print("H200_FINEWEB_SNAPSHOT=verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
