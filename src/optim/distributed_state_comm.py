"""Distributed transport for optimizer state-derived Muon payloads.

DDP already synchronizes gradients during backward.  This module implements a
separate, explicit communication path used by state-sharded optimizers: each
rank owns a row shard of the persistent state, computes its local pre-NS
payload, and all-gathers those rows before the replicated Newton-Schulz update.
"""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch
import torch.distributed as dist

from .fp8_state import (
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)


@dataclass(frozen=True)
class RowShard:
    tensor: torch.Tensor
    original_rows: int


class DistributedStateCommunicator:
    """Shard optimizer state by rows and reconstruct update payloads."""

    def __init__(
        self,
        process_group=None,
        *,
        enabled: bool = False,
        wire_dtype: str = "auto",
        qargs=None,
        profile: bool = False,
    ) -> None:
        if wire_dtype not in {"auto", "bfloat16", "fp8"}:
            raise ValueError(
                "wire_dtype must be one of: auto, bfloat16, fp8; "
                f"got {wire_dtype!r}"
            )
        if wire_dtype == "fp8" and qargs is None:
            raise ValueError("FP8 optimizer-state wire format requires qargs")

        self.process_group = process_group
        self.requested = enabled
        self.wire_dtype = wire_dtype
        self.qargs = qargs
        self.profile_enabled = profile
        self._step_profile: Dict[str, float] = {}
        self._last_profile: Dict[str, float] = {}
        self._pending_cuda_events = []

    @property
    def enabled(self) -> bool:
        return (
            self.requested
            and dist.is_available()
            and dist.is_initialized()
            and self.world_size > 1
        )

    @property
    def world_size(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 1
        return dist.get_world_size(self.process_group)

    @property
    def rank(self) -> int:
        if not dist.is_available() or not dist.is_initialized():
            return 0
        return dist.get_rank(self.process_group)

    def _use_fp8(self, state_derived: bool) -> bool:
        if not state_derived:
            return False
        if self.wire_dtype == "bfloat16":
            return False
        if self.wire_dtype == "fp8":
            return True
        return self.qargs is not None

    def start_step(self) -> None:
        if not self.profile_enabled:
            return
        self._step_profile = {
            "optimizer_state_comm_ms": 0.0,
            "optimizer_state_encode_ms": 0.0,
            "optimizer_state_decode_ms": 0.0,
            "optimizer_state_payload_bytes": 0.0,
            "optimizer_state_wire_bytes": 0.0,
            "optimizer_state_rx_bytes": 0.0,
            "optimizer_stateless_wire_bytes": 0.0,
            "optimizer_stateful_wire_bytes": 0.0,
            "optimizer_state_collectives": 0.0,
        }
        self._pending_cuda_events = []

    @contextmanager
    def phase(self, name: str, tensor: torch.Tensor) -> Iterator[None]:
        if not self.profile_enabled:
            with nullcontext():
                yield
            return

        if tensor.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.nvtx.range_push(name)
            start.record()
            try:
                yield
            finally:
                end.record()
                torch.cuda.nvtx.range_pop()
                self._pending_cuda_events.append((name, start, end))
        else:
            start_time = time.perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1e3
                key = f"optimizer_state_{name}_ms"
                self._step_profile[key] = self._step_profile.get(key, 0.0) + elapsed_ms

    def finish_step(self) -> None:
        if not self.profile_enabled:
            return
        if self._pending_cuda_events:
            torch.cuda.synchronize()
            for name, start, end in self._pending_cuda_events:
                key = f"optimizer_state_{name}_ms"
                self._step_profile[key] = (
                    self._step_profile.get(key, 0.0) + start.elapsed_time(end)
                )
        self._last_profile = dict(self._step_profile)

    def get_last_profile(self) -> Dict[str, float]:
        return dict(self._last_profile)

    def local_rows(self, tensor: torch.Tensor) -> RowShard:
        """Return this rank's equally-sized row shard, padding the last rows."""
        if tensor.ndim < 2:
            raise ValueError(f"row sharding requires a matrix, got shape {tensor.shape}")
        if not self.enabled:
            return RowShard(tensor=tensor, original_rows=tensor.shape[0])

        rows = tensor.shape[0]
        shard_rows = (rows + self.world_size - 1) // self.world_size
        padded_rows = shard_rows * self.world_size
        if padded_rows != rows:
            padding = tensor.new_zeros((padded_rows - rows, *tensor.shape[1:]))
            tensor = torch.cat((tensor, padding), dim=0)
        start = self.rank * shard_rows
        return RowShard(
            tensor=tensor.narrow(0, start, shard_rows).contiguous(),
            original_rows=rows,
        )

    def gather_rows(
        self,
        shard: RowShard,
        *,
        state_derived: bool,
    ) -> torch.Tensor:
        """All-gather a row shard, optionally using an FP8 wire representation."""
        if not self.enabled:
            return shard.tensor

        local = shard.tensor
        # Distributed Muon's uncompressed reference wire format is BF16 even
        # when the local momentum computation is performed in FP32.
        payload_bytes = local.numel() * 2 * self.world_size
        if self._use_fp8(state_derived):
            full, local_wire_bytes = self._gather_fp8(local)
        else:
            full, local_wire_bytes = self._gather_bfloat16(local)

        if self.profile_enabled:
            global_wire_bytes = local_wire_bytes * self.world_size
            self._step_profile["optimizer_state_payload_bytes"] += payload_bytes
            self._step_profile["optimizer_state_wire_bytes"] += global_wire_bytes
            self._step_profile["optimizer_state_rx_bytes"] += (
                local_wire_bytes * (self.world_size - 1)
            )
            kind = "stateful" if state_derived else "stateless"
            self._step_profile[f"optimizer_{kind}_wire_bytes"] += global_wire_bytes
            self._step_profile["optimizer_state_collectives"] += 1

        return full.narrow(0, 0, shard.original_rows).to(dtype=local.dtype)

    def _all_gather_flat(self, local: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            local.numel() * self.world_size,
            dtype=local.dtype,
            device=local.device,
        )
        with self.phase("comm", local):
            dist.all_gather_into_tensor(output, local.reshape(-1), group=self.process_group)
        return output

    def _gather_bfloat16(self, local: torch.Tensor) -> Tuple[torch.Tensor, int]:
        wire = local.to(torch.bfloat16).contiguous()
        gathered = self._all_gather_flat(wire)
        full_shape = (local.shape[0] * self.world_size, *local.shape[1:])
        return gathered.reshape(full_shape), wire.numel() * wire.element_size()

    def _gather_fp8(self, local: torch.Tensor) -> Tuple[torch.Tensor, int]:
        fp8_state: Dict[str, torch.Tensor] = {}
        with self.phase("encode", local):
            init_fp8_state(fp8_state, "wire", local, self.qargs, order="first")
            quantize_fp8_state_(
                fp8_state,
                "wire",
                local,
                self.qargs,
                signed=True,
            )
            names = ["wire", "scale_wire"]
            if "expand_wire" in fp8_state:
                names.extend(("expand_wire", "sqrt_minmax_wire"))
            part_specs = [
                (name, fp8_state[name].shape, fp8_state[name].dtype)
                for name in names
            ]
            packed = torch.cat(
                [fp8_state[name].contiguous().view(torch.uint8).reshape(-1) for name in names]
            )

        gathered = self._all_gather_flat(packed)
        packed_bytes = packed.numel()
        decoded = []
        with self.phase("decode", local):
            for rank in range(self.world_size):
                rank_bytes = gathered.narrow(0, rank * packed_bytes, packed_bytes)
                offset = 0
                rank_state: Dict[str, torch.Tensor] = {}
                for name, shape, dtype in part_specs:
                    numel = math_prod(shape)
                    nbytes = numel * torch.empty((), dtype=dtype).element_size()
                    rank_state[name] = (
                        rank_bytes.narrow(0, offset, nbytes)
                        .clone()
                        .view(dtype)
                        .reshape(shape)
                    )
                    offset += nbytes
                decoded.append(
                    dequantize_fp8_state(
                        rank_state,
                        "wire",
                        self.qargs,
                        signed=True,
                    ).to(local.dtype)
                )
        return torch.cat(decoded, dim=0), packed_bytes


def math_prod(shape: torch.Size) -> int:
    result = 1
    for value in shape:
        result *= value
    return result
