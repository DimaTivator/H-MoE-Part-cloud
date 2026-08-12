#!/usr/bin/env python3
import argparse
import os
import shutil
import time
from pathlib import Path

import torch
import wandb


def artifact_name(group: str, experiment: str, iteration: int) -> str:
    raw = f"{group}-{experiment}-inter-ckpt-{iteration}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in raw)


def checkpoint_iteration(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return int(payload["itr"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="andrey/fp8-pretrain")
    parser.add_argument("--group", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    api = wandb.Api()
    args.destination.mkdir(parents=True, exist_ok=True)
    pending = set(args.iterations)
    while pending:
        for iteration in sorted(pending):
            target = args.destination / str(iteration)
            main_pt = target / "main.pt"
            if main_pt.exists() and checkpoint_iteration(main_pt) == iteration:
                print(f"checkpoint {iteration} already present", flush=True)
                pending.remove(iteration)
                continue
            name = artifact_name(args.group, args.experiment, iteration)
            reference = f"{args.project}/{name}:latest"
            incoming = args.destination / f".{iteration}.incoming"
            try:
                artifact = api.artifact(reference, type="checkpoint")
                shutil.rmtree(incoming, ignore_errors=True)
                artifact.download(root=str(incoming))
                downloaded = incoming / "main.pt"
                actual = checkpoint_iteration(downloaded)
                if actual != iteration:
                    raise RuntimeError(f"artifact {reference} contains iteration {actual}")
                if target.exists():
                    shutil.rmtree(target)
                os.replace(incoming, target)
                print(f"downloaded {reference} -> {target}", flush=True)
                pending.remove(iteration)
            except Exception as exc:
                print(f"waiting for {reference}: {exc}", flush=True)
        if pending:
            time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
