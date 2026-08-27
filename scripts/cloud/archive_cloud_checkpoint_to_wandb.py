#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import wandb


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)


def validate_checkpoint(directory: Path, expected_iteration: int) -> None:
    main_path = directory / "main.pt"
    worker_path = directory / "worker_0.pt"
    if not main_path.is_file() or not worker_path.is_file():
        raise FileNotFoundError(
            f"incomplete checkpoint: main={main_path.is_file()} "
            f"worker={worker_path.is_file()}"
        )
    checkpoint = torch.load(
        main_path, map_location="cpu", mmap=True, weights_only=False
    )
    torch.load(worker_path, map_location="cpu", weights_only=False)
    iteration = int(checkpoint["itr"])
    if iteration != expected_iteration:
        raise ValueError(
            f"checkpoint iteration mismatch: expected={expected_iteration} "
            f"actual={iteration}"
        )
    print(f"VALIDATED_ITERATION={iteration}")
    print(f"MAIN_BYTES={main_path.stat().st_size}")
    print(f"WORKER_BYTES={worker_path.stat().st_size}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="andrey")
    parser.add_argument("--project", default="fp8-pretrain")
    parser.add_argument("--group", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--expected-iteration", type=int, required=True)
    args = parser.parse_args()

    if args.expected_iteration <= 0:
        raise ValueError("expected iteration must be positive")
    if sanitize(args.group) != args.group or sanitize(args.experiment) != args.experiment:
        raise ValueError("group and experiment may only contain [A-Za-z0-9._-]")
    if args.checkpoint_name != str(args.expected_iteration):
        raise ValueError("checkpoint name must equal expected iteration")

    checkpoint_dir = (
        args.results_dir
        / args.group
        / args.experiment
        / "ckpts"
        / args.checkpoint_name
    )
    validate_checkpoint(checkpoint_dir, args.expected_iteration)

    artifact_name = sanitize(
        f"{args.group}-{args.experiment}-inter-ckpt-{args.expected_iteration}"
    )
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        job_type="checkpoint-archive",
        name=f"{args.experiment}-archive-{args.expected_iteration}",
    )
    artifact = wandb.Artifact(
        name=artifact_name,
        type="checkpoint",
        description="Permanent checkpoint archived from Cloud storage.",
        metadata={
            "iteration": args.expected_iteration,
            "experiment_name": args.experiment,
            "wandb_group": args.group,
        },
    )
    artifact.add_dir(str(checkpoint_dir))
    logged = run.log_artifact(artifact, aliases=[str(args.expected_iteration)])
    logged.wait()
    run.finish()
    print(f"ARCHIVED_ARTIFACT={artifact_name}:{args.expected_iteration}")
    print(f"CLOUD_CHECKPOINT_PRESERVED={checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
