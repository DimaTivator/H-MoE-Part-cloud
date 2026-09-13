#!/usr/bin/env python3
"""Microbenchmark Muon state-derived all-gather on representative LLM matrices."""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from optim.distributed_state_comm import DistributedStateCommunicator


SHAPE_PRESETS = {
    "tiny": [(256, 256), (768, 256), (256, 768)],
    "llama_257m": [(1024, 1024), (3072, 1024), (1024, 3072)],
    "llama_7b": [(4096, 4096), (11008, 4096), (4096, 11008)],
}
METHODS = (
    "muon",
    "muon_fp8_states",
    "frugal_muon_muon",
    "frugal_muon_muon_fp8_states",
)
RESULT_PREFIX = "[STATE COMM MICROBENCH RESULT] "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=SHAPE_PRESETS, default="llama_257m")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=100)
    parser.add_argument(
        "--fp8-expansion",
        choices=["false", "expand"],
        default="expand",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_qargs(expansion: str) -> SimpleNamespace:
    return SimpleNamespace(
        qgroup_size=128,
        first_order_bit="E4M3",
        second_order_bit="E4M3",
        first_order_expansion=expansion,
        second_order_expansion=expansion,
        expand_min=16,
    )


def distributed_max(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.cpu().tolist()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def benchmark(
    method: str,
    shape: tuple[int, int],
    density: float,
    warmup_steps: int,
    measure_steps: int,
    device: torch.device,
    fp8_expansion: str,
) -> dict[str, float | int | str]:
    fp8 = method.endswith("fp8_states")
    frugal = method.startswith("frugal")
    communicator = DistributedStateCommunicator(
        enabled=True,
        wire_dtype="fp8" if fp8 else "bfloat16",
        qargs=make_qargs(fp8_expansion) if fp8 else None,
        profile=True,
    )
    source = torch.randn(shape, dtype=torch.bfloat16, device=device)
    if frugal:
        active_columns = max(1, int(shape[1] * density))
        active = source[:, :active_columns].contiguous()
        inactive = source[:, active_columns:].contiguous()
    else:
        active = source
        inactive = None

    total_times = []
    profiles = []
    for iteration in range(warmup_steps + measure_steps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        communicator.start_step()
        communicator.gather_rows(
            communicator.local_rows(active),
            state_derived=True,
        )
        if inactive is not None and inactive.numel() > 0:
            communicator.gather_rows(
                communicator.local_rows(inactive),
                state_derived=False,
            )
        communicator.finish_step()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        if iteration >= warmup_steps:
            total_times.append(elapsed_ms)
            profiles.append(communicator.get_last_profile())

    total_times = distributed_max(total_times, device)
    result: dict[str, float | int | str] = {
        "method": method,
        "rows": shape[0],
        "columns": shape[1],
        "world_size": dist.get_world_size(),
        "density": density if frugal else 1.0,
        "measure_steps": measure_steps,
        "total_ms_mean": sum(total_times) / len(total_times),
        "total_ms_p50": percentile(total_times, 0.50),
        "total_ms_p95": percentile(total_times, 0.95),
    }
    profile_keys = sorted({key for profile in profiles for key in profile})
    for key in profile_keys:
        values = distributed_max(
            [float(profile.get(key, 0.0)) for profile in profiles],
            device,
        )
        result[f"{key}_mean"] = sum(values) / len(values)
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 < args.density <= 1.0:
        raise SystemExit("density must be in (0, 1]")
    if args.warmup_steps < 0 or args.measure_steps <= 0:
        raise SystemExit("warmup_steps must be >= 0 and measure_steps must be > 0")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)

    for shape in SHAPE_PRESETS[args.preset]:
        for method in args.methods:
            result = benchmark(
                method,
                shape,
                args.density,
                args.warmup_steps,
                args.measure_steps,
                device,
                args.fp8_expansion,
            )
            if dist.get_rank() == 0:
                print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
