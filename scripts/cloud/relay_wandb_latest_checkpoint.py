#!/usr/bin/env python3
import argparse
import os
import shutil
import time
from pathlib import Path

import torch
import wandb


def sanitize_artifact_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)


def artifact_name(group: str, experiment: str) -> str:
    return sanitize_artifact_name(f"{group}-{experiment}-latest-ckpt")


def checkpoint_iteration(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return int(payload["itr"])


def validate_checkpoint(directory: Path) -> int:
    main_pt = directory / "main.pt"
    worker_pt = directory / "worker_0.pt"
    if not main_pt.is_file() or not worker_pt.is_file():
        raise RuntimeError(f"incomplete checkpoint in {directory}")
    iteration = checkpoint_iteration(main_pt)
    if iteration <= 0:
        raise RuntimeError(f"invalid checkpoint iteration {iteration}")
    return iteration


def install_checkpoint(incoming: Path, target: Path, previous: Path) -> None:
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
    parser.add_argument("--project", default="andrey/fp8-pretrain")
    parser.add_argument("--group", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--final-iteration", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    if args.final_iteration <= 0 or args.poll_seconds <= 0:
        raise ValueError("final iteration and poll interval must be positive")

    api = wandb.Api()
    args.destination.mkdir(parents=True, exist_ok=True)
    target = args.destination / "latest"
    incoming = args.destination / ".latest.incoming"
    previous = args.destination / ".latest.previous"
    current_iteration = validate_checkpoint(target) if target.exists() else -1
    reference = f"{args.project}/{artifact_name(args.group, args.experiment)}:latest"

    while current_iteration < args.final_iteration:
        try:
            artifact = api.artifact(reference, type="checkpoint")
            remote_iteration = int(artifact.metadata["iteration"])
            if remote_iteration > current_iteration:
                shutil.rmtree(incoming, ignore_errors=True)
                artifact.download(root=str(incoming))
                downloaded_iteration = validate_checkpoint(incoming)
                if downloaded_iteration != remote_iteration:
                    raise RuntimeError(
                        f"artifact metadata says {remote_iteration}, "
                        f"checkpoint contains {downloaded_iteration}"
                    )
                install_checkpoint(incoming, target, previous)
                current_iteration = downloaded_iteration
                print(
                    f"RELAYED_ITERATION={current_iteration} TARGET={target}",
                    flush=True,
                )
            else:
                print(f"WAITING_CURRENT_ITERATION={current_iteration}", flush=True)
        except Exception as exc:
            print(f"WAITING_FOR={reference} ERROR={exc}", flush=True)
            shutil.rmtree(incoming, ignore_errors=True)

        if current_iteration < args.final_iteration:
            time.sleep(args.poll_seconds)

    print(f"FINAL_RELAY_ITERATION={current_iteration}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
