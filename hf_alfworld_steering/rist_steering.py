from __future__ import annotations

import torch


def resolve_blocks(model):
    m = model
    if hasattr(m, "base_model"):
        m = m.base_model

    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers

    if hasattr(m, "model") and hasattr(m.model, "model") and hasattr(m.model.model, "layers"):
        return m.model.model.layers

    raise RuntimeError(f"Cannot resolve transformer blocks for {type(model)}")


class GatedActivationSteerer:
    """
    Dynamic risk-guided steering.

    Computes projection of hidden state onto hack direction.
    If projection > tau, subtract alpha * hack_direction.
    """

    def __init__(
        self,
        model,
        directions: dict[int, torch.Tensor],
        alpha: float,
        tau: float = 0.0,
        mode: str = "gated",  # "always" or "gated"
        token_scope: str = "last",  # "last" or "all"
    ):
        self.model = model
        self.directions = directions
        self.layers = list(directions.keys())
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.mode = mode
        self.token_scope = token_scope
        self.handles = []

        self.risk_scores = []
        self.num_steered = 0
        self.num_seen = 0

    def _hook(self, layer_idx):
        direction = self.directions[layer_idx].to(next(self.model.parameters()).device)
        direction = direction / (direction.norm() + 1e-8)

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out

            # h: [batch, seq, hidden]
            h_float = h.float()

            if self.token_scope == "last":
                probe_h = h_float[:, -1:, :]
            else:
                probe_h = h_float

            risk = torch.matmul(probe_h, direction.float())
            risk_scalar = risk.mean().detach().float().item()

            self.risk_scores.append(
                {
                    "layer": layer_idx,
                    "risk": risk_scalar,
                    "tau": self.tau,
                    "steered": False,
                }
            )

            should_steer = self.mode == "always" or risk_scalar > self.tau

            if should_steer and self.alpha != 0.0:
                self.num_steered += 1
                if self.token_scope == "last":
                    h = h.clone()
                    h[:, -1:, :] = h[:, -1:, :] - self.alpha * direction.view(1, 1, -1).to(h.dtype)
                else:
                    h = h - self.alpha * direction.view(1, 1, -1).to(h.dtype)

                self.risk_scores[-1]["steered"] = True

            self.num_seen += 1

            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        return hook

    def __enter__(self):
        blocks = resolve_blocks(self.model)

        for l in self.layers:
            self.handles.append(blocks[l].register_forward_hook(self._hook(l)))

        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []

    def summary(self):
        risks = [x["risk"] for x in self.risk_scores]
        return {
            "steering_alpha": self.alpha,
            "steering_tau": self.tau,
            "steering_mode": self.mode,
            "steering_num_seen": self.num_seen,
            "steering_num_steered": self.num_steered,
            "steering_rate": self.num_steered / max(self.num_seen, 1),
            "steering_risk_mean": float(sum(risks) / len(risks)) if risks else None,
            "steering_risk_max": float(max(risks)) if risks else None,
        }