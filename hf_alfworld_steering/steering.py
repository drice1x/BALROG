from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict
import torch


class ActivationCollector:
    def __init__(self, model, layers: list[int]):
        self.model = model
        self.layers = layers
        self.cache = defaultdict(list)
        self.handles = []
        #self.layers = list(directions.keys())

    def _hook(self, layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.cache[layer_idx].append(h.detach().float().cpu())
        return hook

    def __enter__(self):
        # robustly resolve transformer blocks across
        # HF base model + PEFT wrapped model

        model_ref = self.model

        # unwrap PEFT if present
        if hasattr(model_ref, "base_model"):
            model_ref = model_ref.base_model

        # common Qwen/Llama paths
        if hasattr(model_ref, "model"):
            if hasattr(model_ref.model, "layers"):
                blocks = model_ref.model.layers
            elif hasattr(model_ref.model, "model") and hasattr(model_ref.model.model, "layers"):
                blocks = model_ref.model.model.layers
            else:
                raise RuntimeError(
                    f"Could not find transformer layers. model structure: {type(model_ref)}"
                )
        else:
            raise RuntimeError(
                f"Unexpected model structure: {type(model_ref)}"
            )

        for l in self.layers:
            h = blocks[l].register_forward_hook(self._hook(l))
            self.handles.append(h)

        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()

    def mean_by_layer(self):
        out = {}
        for l, tensors in self.cache.items():
            # concatenate over generation calls, use last-token activations
            vals = []
            for t in tensors:
                vals.append(t[:, -1, :])
            out[l] = torch.cat(vals, dim=0).mean(dim=0)
        return out


class ActivationSteerer:
    def __init__(self, model, directions: dict[int, torch.Tensor], alpha: float):
        self.model = model
        self.directions = directions
        self.alpha = alpha
        self.handles = []
        self.layers = list(directions.keys())

    def _hook(self, layer_idx):
        direction = self.directions[layer_idx].to(next(self.model.parameters()).device)
        direction = direction / (direction.norm() + 1e-8)

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h - self.alpha * direction.view(1, 1, -1).to(h.dtype)
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        return hook

    def __enter__(self):
        # robustly resolve transformer blocks across
        # HF base model + PEFT wrapped model

        model_ref = self.model

        # unwrap PEFT if present
        if hasattr(model_ref, "base_model"):
            model_ref = model_ref.base_model

        # common Qwen/Llama paths
        if hasattr(model_ref, "model"):
            if hasattr(model_ref.model, "layers"):
                blocks = model_ref.model.layers
            elif hasattr(model_ref.model, "model") and hasattr(model_ref.model.model, "layers"):
                blocks = model_ref.model.model.layers
            else:
                raise RuntimeError(
                    f"Could not find transformer layers. model structure: {type(model_ref)}"
                )
        else:
            raise RuntimeError(
                f"Unexpected model structure: {type(model_ref)}"
            )

        for l in self.layers:
            h = blocks[l].register_forward_hook(self._hook(l))
            self.handles.append(h)

        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()