#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path

import torch
import wandb


def validate_checkpoint(directory: Path, expected_iteration: int) -> None:
    main_path = directory / "main.pt"
    worker_path = directory / "worker_0.pt"
    if not main_path.is_file() or not worker_path.is_file():
        raise RuntimeError(f"incomplete checkpoint in {directory}")
    checkpoint = torch.load(
        main_path, map_location="cpu", mmap=True, weights_only=False
    )
    torch.load(worker_path, map_location="cpu", weights_only=False)
    iteration = int(checkpoint["itr"])
    if iteration != expected_iteration:
        raise RuntimeError(
            f"checkpoint iteration mismatch: expected={expected_iteration} "
            f"actual={iteration}"
        )


def install(incoming: Path, target: Path, previous: Path) -> None:
    shutil.rmtree(previous, ignore_errors=True)
    if target.exists():
        os.replace(target, previous)
    try:
        os.replace(incoming, target)
    except BaseException:
        if previous.exists() and not target.exists():
            os.replace(previous, target)
        raise
    shutil.rmtree(previous, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    args = parser.parse_args()

    if args.expected_iteration <= 0:
        raise ValueError("expected iteration must be positive")
    if args.checkpoint_name != str(args.expected_iteration):
        raise ValueError("checkpoint name must equal expected iteration")

    args.destination.mkdir(parents=True, exist_ok=True)
    target = args.destination / args.checkpoint_name
    incoming = args.destination / f".{args.checkpoint_name}.incoming"
    previous = args.destination / f".{args.checkpoint_name}.previous"
    shutil.rmtree(incoming, ignore_errors=True)

    artifact = wandb.Api().artifact(args.artifact, type="checkpoint")
    remote_iteration = int(artifact.metadata["iteration"])
    if remote_iteration != args.expected_iteration:
        raise RuntimeError(
            f"artifact iteration mismatch: expected={args.expected_iteration} "
            f"actual={remote_iteration}"
        )
    artifact.download(root=str(incoming))
    validate_checkpoint(incoming, args.expected_iteration)
    install(incoming, target, previous)
    print(f"RELAYED_ITERATION={args.expected_iteration}")
    print(f"TARGET={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
