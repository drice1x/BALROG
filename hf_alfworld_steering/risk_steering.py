from __future__ import annotations

from typing import Any

import torch


def resolve_blocks(model) -> Any:
    model_ref = model

    # PEFT models commonly expose the wrapped model through one of these attrs.
    for attr in ("base_model", "model"):
        nested = getattr(model_ref, attr, None)
        if nested is not None and hasattr(nested, "model"):
            model_ref = nested
            break

    candidates = [
        ("model.layers", lambda m: m.model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("layers", lambda m: m.layers),
        ("transformer.h", lambda m: m.transformer.h),
    ]
    for _name, getter in candidates:
        try:
            blocks = getter(model_ref)
            if blocks is not None:
                return blocks
        except AttributeError:
            continue

    raise RuntimeError(f"Could not resolve transformer blocks for model type {type(model)}")


class GatedActivationSteerer:
    def __init__(
        self,
        model,
        directions: dict[int, torch.Tensor],
        alpha: float,
        mode: str = "always",
        tau: float = 0.0,
        token_scope: str = "last",
    ):
        self.model = model
        self.directions = directions or {}
        self.alpha = float(alpha)
        self.mode = mode
        self.tau = float(tau)
        self.token_scope = token_scope
        self.handles = []

        self.steering_num_seen = 0
        self.steering_num_steered = 0
        self._risk_sum = 0.0
        self.steering_risk_max = float("-inf")

    def _normalize_direction(self, layer_idx: int, device, dtype):
        direction = self.directions[layer_idx].to(device=device, dtype=torch.float32)
        direction = direction / (direction.norm() + 1e-8)
        return direction.to(dtype=dtype)

    def _select_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.token_scope == "all":
            return hidden
        if self.token_scope == "last":
            return hidden[:, -1:, :]
        raise ValueError(f"Unsupported token_scope: {self.token_scope}")

    def _hook(self, layer_idx: int):
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            direction = self._normalize_direction(layer_idx, hidden.device, hidden.dtype)
            selected = self._select_tokens(hidden)
            risk_tensor = torch.matmul(selected.float(), direction.float())
            risk = float(risk_tensor.mean().item())

            self.steering_num_seen += 1
            self._risk_sum += risk
            self.steering_risk_max = max(self.steering_risk_max, risk)

            should_steer = self.mode == "always" or (self.mode == "gated" and risk > self.tau)
            if should_steer:
                self.steering_num_steered += 1
                if self.token_scope == "all":
                    hidden = hidden - self.alpha * direction.view(1, 1, -1)
                else:
                    hidden = hidden.clone()
                    hidden[:, -1:, :] = hidden[:, -1:, :] - self.alpha * direction.view(1, 1, -1)

            if isinstance(out, tuple):
                return (hidden,) + out[1:]
            return hidden

        return hook

    def __enter__(self):
        blocks = resolve_blocks(self.model)
        for layer_idx in sorted(self.directions):
            self.handles.append(blocks[layer_idx].register_forward_hook(self._hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @property
    def steering_rate(self) -> float:
        if self.steering_num_seen == 0:
            return 0.0
        return self.steering_num_steered / self.steering_num_seen

    @property
    def steering_risk_mean(self) -> float:
        if self.steering_num_seen == 0:
            return 0.0
        return self._risk_sum / self.steering_num_seen

    def summary(self) -> dict[str, float | int | str]:
        return {
            "steering_mode": self.mode,
            "steering_alpha": self.alpha,
            "steering_tau": self.tau,
            "steering_token_scope": self.token_scope,
            "steering_num_seen": self.steering_num_seen,
            "steering_num_steered": self.steering_num_steered,
            "steering_rate": self.steering_rate,
            "steering_risk_mean": self.steering_risk_mean,
            "steering_risk_max": 0.0 if self.steering_num_seen == 0 else self.steering_risk_max,
        }
