import os
import math
import time
from contextlib import contextmanager

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import (
    init_process_group,
    destroy_process_group,
    get_world_size,
    barrier,
    all_reduce,
)

from .backend import DistributedBackend


class DataParallelDistributedBackend(DistributedBackend):

    def __init__(self, args):
        self.rank = int(os.environ.get("RANK", -1))
        assert self.rank != -1, "DDP backend can not be used without rank"
        assert "cuda" in args.device, "DDP backend can not be used on non-CUDA devices"
        init_process_group(backend=args.distributed_backend)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.profile_communication = getattr(args, "optimizer_comm_profile", False)
        self._profile_active = False
        self._step_comm_profile = {}
        self._last_comm_profile = {}

    def get_adjusted_args_for_process(self, args):
        effective_batch_size = args.batch_size * args.acc_steps
        world_size = self.get_world_size()
        if effective_batch_size % world_size != 0:
            raise ValueError(
                f"Effective batch size "
                "{effective_batch_size} is not divisible "
                "by the world size {world_size}."
            )
        acc_steps_div = math.gcd(args.acc_steps, world_size)
        args.acc_steps = args.acc_steps // acc_steps_div
        args.batch_size = args.batch_size // (world_size // acc_steps_div)
        args.device = f"cuda:{self.local_rank}"
        args.seed = args.seed + self.local_rank
        args.data_seed = args.data_seed
        return args

    def transform_model(self, model, **kwargs):
        model = DDP(model, device_ids=[self.local_rank], **kwargs)
        if self.profile_communication and self.get_world_size() > 1:
            model.register_comm_hook(self, _profiled_allreduce_hook)
        return model

    @contextmanager
    def get_context_for_microstep_forward(
        self, model, microstep_idx, gradient_accumulation_steps
    ):
        model.require_backward_grad_sync = (
            microstep_idx == gradient_accumulation_steps - 1
        )
        yield

    def is_master_process(self) -> bool:
        return self.rank == 0

    def get_raw_model(self, model):
        return model.module

    def translate_model_parameter_name_for_node(self, parameter_name):
        return [f"module.{parameter_name}"]

    def get_world_size(self):
        return get_world_size()

    def finalize(self):
        destroy_process_group()

    def barrier(self):
        barrier()

    def start_step_profile(self):
        if not self.profile_communication:
            return
        self._profile_active = True
        self._step_comm_profile = {
            "ddp_gradient_payload_bytes": 0.0,
            "ddp_gradient_ring_traffic_bytes_per_rank": 0.0,
            "ddp_gradient_collectives": 0.0,
            "_first_start": None,
            "_last_end": None,
        }

    def finish_step_profile(self):
        if not self.profile_communication:
            return
        self._profile_active = False
        first = self._step_comm_profile.pop("_first_start")
        last = self._step_comm_profile.pop("_last_end")
        self._step_comm_profile["ddp_gradient_comm_span_ms"] = (
            (last - first) * 1e3 if first is not None and last is not None else 0.0
        )
        self._last_comm_profile = dict(self._step_comm_profile)

    def get_last_comm_profile(self):
        return dict(self._last_comm_profile)


def _profiled_allreduce_hook(state: DataParallelDistributedBackend, bucket):
    tensor = bucket.buffer()
    if state._profile_active:
        now = time.perf_counter()
        if state._step_comm_profile["_first_start"] is None:
            state._step_comm_profile["_first_start"] = now
        payload_bytes = tensor.numel() * tensor.element_size()
        world_size = get_world_size()
        state._step_comm_profile["ddp_gradient_payload_bytes"] += payload_bytes
        state._step_comm_profile["ddp_gradient_ring_traffic_bytes_per_rank"] += (
            2.0 * payload_bytes * (world_size - 1) / world_size
        )
        state._step_comm_profile["ddp_gradient_collectives"] += 1

    tensor.div_(get_world_size())
    future = all_reduce(tensor, async_op=True).get_future()

    def _done(result):
        if state._profile_active:
            state._step_comm_profile["_last_end"] = time.perf_counter()
        return result.value()[0]

    return future.then(_done)
