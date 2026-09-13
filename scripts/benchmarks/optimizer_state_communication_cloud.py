#!/usr/bin/env python3
"""Run optimizer-state communication benchmarks under an mlsub MPI launch."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from optimizer_state_communication import MODEL_CONFIGS, METHODS, method_args


RESULT_PREFIX = "[STEP TIME BENCH RESULT] "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_CONFIGS, default=["500M", "1B"])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--local-batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--update-gap", type=int, default=50)
    parser.add_argument("--fp8-expansion", choices=["false", "expand"], default="expand")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--eval-cache-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def distributed_environment() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    )
    return world_size, rank, local_rank


def build_command(
    args: argparse.Namespace,
    model: str,
    method: str,
    world_size: int,
    repetition: int,
) -> list[str]:
    cfg = MODEL_CONFIGS[model]
    total_steps = args.warmup_steps + args.measure_steps
    command = [
        sys.executable,
        "src/main.py",
        "--distributed-backend", "nccl",
        "--experiment-name", f"state_comm_{model}_{method}_p{world_size}_r{repetition}",
        "--seed", "1234",
        "--data-seed", "1337",
        "--dataset", "c4",
        "--datasets-dir", args.datasets_dir,
        "--eval-cache-dir", args.eval_cache_dir,
        "--sequence-length", str(args.sequence_length),
        "--workers", str(args.workers),
        "--model", "llama",
        "--n-layer", str(cfg["n_layer"]),
        "--n-embd", str(cfg["n_embd"]),
        "--n-head", str(cfg["n_head"]),
        "--multiple-of", "256",
        "--dtype", "bfloat16",
        "--lr", "1e-3",
        "--weight-decay", "0.1",
        "--grad-clip", "1.0",
        "--scheduler", "none",
        "--warmup-steps", "0",
        "--iterations", str(total_steps),
        "--batch-size", str(args.local_batch_size * world_size),
        "--eval-batch-size", "1",
        "--acc-steps", "1",
        "--eval-interval", str(total_steps + 1),
        "--eval-batches", "1",
        "--log-interval", "0",
        "--no-local-save",
    ]
    command.extend(
        value.format(density=args.density, update_gap=args.update_gap)
        for value in method_args(args, method)
    )
    return command


def append_row(output_dir: Path, row: dict[str, Any]) -> None:
    jsonl_path = output_dir / "results.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    with jsonl_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    keys = sorted({key for item in rows for key in item})
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def method_order(methods: list[str], repetition: int) -> list[str]:
    if not methods:
        return []
    shift = repetition % len(methods)
    ordered = methods[shift:] + methods[:shift]
    if repetition % 2:
        ordered.reverse()
    return ordered


def run_one(
    args: argparse.Namespace,
    model: str,
    method: str,
    world_size: int,
    rank: int,
    repetition: int,
) -> bool:
    name = f"{model}_{method}_p{world_size}_r{repetition}"
    log_path = args.output_dir / f"{name}.rank{rank}.log"
    command = build_command(args, model, method, world_size, repetition)
    env = os.environ.copy()
    env.update(
        {
            "STEP_TIME_BENCH": "1",
            "STEP_TIME_WARMUP_STEPS": str(args.warmup_steps),
            "STEP_TIME_MEASURE_STEPS": str(args.measure_steps),
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
        }
    )

    if rank == 0:
        print(f"[RUN] {name}", flush=True)
    metric = None
    tail: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_stream.write(line)
            if rank == 0:
                print(line, end="", flush=True)
                if RESULT_PREFIX in line:
                    metric = json.loads(line.split(RESULT_PREFIX, 1)[1])
            tail.append(line.rstrip())
            tail = tail[-80:]
        return_code = process.wait()

    if rank == 0:
        cfg = MODEL_CONFIGS[model]
        row: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": os.environ.get("RUN_ID", "unknown"),
            "model": model,
            "n_layer": cfg["n_layer"],
            "n_embd": cfg["n_embd"],
            "n_head": cfg["n_head"],
            "method": method,
            "repetition": repetition,
            "world_size": world_size,
            "local_batch_size": args.local_batch_size,
            "sequence_length": args.sequence_length,
            "density": args.density if "frugal" in method else 1.0,
            "gradient_sync": "ddp_allreduce",
            "optimizer_state_sync": "row_shard_allgather",
            "parameter_sync": "none_replicated_parameters",
            "return_code": return_code,
            "log_file": str(log_path),
        }
        if metric is not None and return_code == 0:
            row.update(metric)
            row["status"] = "ok"
        else:
            row["status"] = "failed"
            row["error"] = "\n".join(tail)[-4000:]
        append_row(args.output_dir, row)
    return return_code == 0


def main() -> int:
    args = parse_args()
    world_size, rank, local_rank = distributed_environment()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(
            f"BENCHMARK_CONFIG world_size={world_size} models={args.models} "
            f"methods={args.methods} repeats={args.repeats} "
            f"warmup={args.warmup_steps} measure={args.measure_steps}",
            flush=True,
        )
    for repetition in range(args.repeats):
        for model in args.models:
            for method in method_order(list(args.methods), repetition):
                if not run_one(args, model, method, world_size, rank, repetition):
                    return 1
    if rank == 0:
        print(f"BENCHMARK_COMPLETE={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
