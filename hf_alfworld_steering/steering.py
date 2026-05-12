from __future__ import annotations

from collections import defaultdict
import torch

from risk_steering import resolve_blocks


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
        blocks = resolve_blocks(self.model)

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
        blocks = resolve_blocks(self.model)

        for l in self.layers:
            h = blocks[l].register_forward_hook(self._hook(l))
            self.handles.append(h)

        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
