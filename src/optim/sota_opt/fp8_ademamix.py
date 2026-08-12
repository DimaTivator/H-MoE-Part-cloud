import math

import torch

from optim.fp8_state import (
    FP8StateDictMixin,
    dequantize_fp8_state,
    init_fp8_state,
    quantize_fp8_state_,
)

from .ademamix import AdEMAMix, linear_hl_warmup_scheduler, linear_warmup_scheduler


class FP8AdEMAMix(FP8StateDictMixin, AdEMAMix):
    def __init__(self, params, qargs, **kwargs):
        super().__init__(params, **kwargs)
        self.qargs = qargs

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    if beta1 != 0.0:
                        init_fp8_state(state, "exp_avg_fast", param, self.qargs, order="first")
                    else:
                        state["exp_avg_fast"] = None
                    init_fp8_state(state, "exp_avg_slow", param, self.qargs, order="first")
                    init_fp8_state(state, "exp_avg_sq", param, self.qargs, order="second")

                grad_fp32 = grad.to(torch.float32)
                if beta1 != 0.0:
                    exp_avg_fast = dequantize_fp8_state(
                        state,
                        "exp_avg_fast",
                        self.qargs,
                        signed=True,
                    )
                else:
                    exp_avg_fast = grad_fp32
                exp_avg_slow = dequantize_fp8_state(
                    state,
                    "exp_avg_slow",
                    self.qargs,
                    signed=True,
                )
                exp_avg_sq = dequantize_fp8_state(
                    state,
                    "exp_avg_sq",
                    self.qargs,
                    signed=False,
                )

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                alpha = (
                    linear_warmup_scheduler(
                        state["step"],
                        alpha_end=alpha_final,
                        alpha_start=0,
                        warmup=alpha_warmup,
                    )
                    if alpha_warmup is not None
                    else alpha_final
                )
                beta3 = (
                    linear_hl_warmup_scheduler(
                        state["step"],
                        beta_end=beta3_final,
                        beta_start=beta1,
                        warmup=beta3_warmup,
                    )
                    if beta3_warmup is not None
                    else beta3_final
                )

                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
                exp_avg_slow.mul_(beta3).add_(grad_fp32, alpha=1 - beta3)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)

                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                update = (exp_avg_fast.div(bias_correction1) + alpha * exp_avg_slow) / denom
                update.add_(param.to(torch.float32), alpha=weight_decay)
                param.add_(update.to(dtype=param.dtype), alpha=-lr)

                if beta1 != 0.0:
                    quantize_fp8_state_(
                        state,
                        "exp_avg_fast",
                        exp_avg_fast,
                        self.qargs,
                        signed=True,
                    )
                quantize_fp8_state_(
                    state,
                    "exp_avg_slow",
                    exp_avg_slow,
                    self.qargs,
                    signed=True,
                )
                quantize_fp8_state_(
                    state,
                    "exp_avg_sq",
                    exp_avg_sq,
                    self.qargs,
                    signed=False,
                )

        return loss
