import math

import torch


class AdEMAMix(torch.optim.Optimizer):
    """Minimal AdEMAMix implementation for the Stage 4 Megatron benchmark."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),
        alpha=8.0,
        eps=1e-8,
        weight_decay=0.0,
        beta3_warmup_steps=None,
        alpha_warmup_steps=None,
    ):
        if len(betas) != 3:
            raise ValueError("AdEMAMix expects three beta values")
        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            beta3_warmup_steps=beta3_warmup_steps,
            alpha_warmup_steps=alpha_warmup_steps,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _scheduled_beta3(beta1, beta3, step, warmup_steps):
        if warmup_steps is None or step >= warmup_steps:
            return beta3
        start = math.log(0.5) / math.log(beta1) - 1.0
        end = math.log(0.5) / math.log(beta3) - 1.0
        half_life = start + (end - start) * step / warmup_steps
        return 0.5 ** (1.0 / (half_life + 1.0))

    @staticmethod
    def _scheduled_alpha(alpha, step, warmup_steps):
        if warmup_steps is None or step >= warmup_steps:
            return alpha
        return alpha * step / warmup_steps

    @torch.no_grad()
    def _init_group(self, group, skip_non_grad_params=True):
        for parameter in group["params"]:
            if skip_non_grad_params and parameter.grad is None:
                continue
            state = self.state[parameter]
            if not state:
                state["step"] = 0
                state["exp_avg_fast"] = torch.zeros_like(
                    parameter, dtype=torch.float32
                )
                state["exp_avg_slow"] = torch.zeros_like(
                    parameter, dtype=torch.float32
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    parameter, dtype=torch.float32
                )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            self._init_group(group)
            beta1, beta2, beta3_final = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad.float()
                state = self.state[parameter]
                state["step"] += 1
                step = state["step"]
                beta3 = self._scheduled_beta3(
                    beta1, beta3_final, step, group["beta3_warmup_steps"]
                )
                alpha = self._scheduled_alpha(
                    group["alpha"], step, group["alpha_warmup_steps"]
                )

                fast = state["exp_avg_fast"]
                slow = state["exp_avg_slow"]
                square = state["exp_avg_sq"]
                fast.lerp_(grad, 1.0 - beta1)
                slow.lerp_(grad, 1.0 - beta3)
                square.lerp_(grad.square(), 1.0 - beta2)

                update = fast.div(1.0 - beta1**step).add(slow, alpha=alpha)
                denominator = square.div(1.0 - beta2**step).sqrt_().add_(
                    group["eps"]
                )
                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(update, denominator, value=-group["lr"])

        return loss
