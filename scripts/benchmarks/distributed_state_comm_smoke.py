#!/usr/bin/env python3
"""Torchrun smoke test for BF16 and FP8 optimizer-state transport."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from optim.distributed_state_comm import DistributedStateCommunicator


def qargs(expansion: str = "false") -> SimpleNamespace:
    return SimpleNamespace(
        qgroup_size=8,
        first_order_bit="E4M3",
        second_order_bit="E4M3",
        first_order_expansion=expansion,
        second_order_expansion=expansion,
        expand_min=16,
    )


def check(wire_dtype: str, device: torch.device, expansion: str = "false") -> float:
    codec_args = qargs(expansion) if wire_dtype == "fp8" else None
    communicator = DistributedStateCommunicator(
        enabled=True,
        wire_dtype=wire_dtype,
        qargs=codec_args,
        profile=True,
    )
    full = torch.linspace(-2.0, 2.0, 77, device=device).reshape(7, 11)
    shard = communicator.local_rows(full)
    communicator.start_step()
    reconstructed = communicator.gather_rows(shard, state_derived=True)
    communicator.finish_step()
    tolerance = 0.15 if wire_dtype == "fp8" else 0.02
    error = (reconstructed.float() - full.float()).abs().max().item()
    if error > tolerance:
        raise AssertionError(
            f"{wire_dtype} reconstruction error {error:.6f} > {tolerance:.6f}"
        )
    return error


def main() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    errors = {
        "bfloat16": check("bfloat16", device),
        "fp8": check("fp8", device),
        "fp8_expand": check("fp8", device, expansion="expand"),
    }
    if dist.get_rank() == 0:
        print(json.dumps({"status": "ok", "world_size": dist.get_world_size(), **errors}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
