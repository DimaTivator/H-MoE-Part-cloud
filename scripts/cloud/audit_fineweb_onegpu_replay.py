#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.fineweb import build_fineweb_readers  # noqa: E402
from src.data.fineweb_replay import blocks_sha256  # noqa: E402


def tokenizer_factory():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024
    return tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.batches <= 0:
        parser.error("--batches must be positive")

    reader_args = SimpleNamespace(
        datasets_dir=args.datasets_dir,
        sequence_length=1024,
        eval_batches=32,
        eval_batch_size=32,
        data_seed=1337,
        batch_size=32,
        workers=args.workers,
        tokenizer="gpt2",
        device="cpu",
        fineweb_replay_world_size=2,
    )
    tokenizer = tokenizer_factory()
    readers = build_fineweb_readers(
        reader_args,
        tokenizer=tokenizer,
        tokenizer_factory=tokenizer_factory,
        verbose=True,
    )

    validation_digest = blocks_sha256(readers["val"].blocks, dtype="<u4")
    batch_digests = []
    for microstep in range(args.batches):
        x, y = readers["train"].sample_batch()
        blocks = torch.cat((x, y[:, -1:]), dim=1)
        digest = blocks_sha256(blocks)
        batch_digests.append(digest)
        print(
            f"FINEWEB_ONEGPU_BATCH_SHA256 microstep={microstep} sha256={digest}",
            flush=True,
        )

    result = {
        "format": "fineweb_onegpu_replay_audit_v1",
        "datasets_dir": args.datasets_dir,
        "data_seed": 1337,
        "source_world_size": 2,
        "source_batch_size": 16,
        "onegpu_batch_size": 32,
        "sequence_length": 1024,
        "validation_sha256_uint32": validation_digest,
        "batch_sha256_uint16": batch_digests,
    }
    print(f"FINEWEB_ONEGPU_REPLAY_AUDIT={json.dumps(result, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
