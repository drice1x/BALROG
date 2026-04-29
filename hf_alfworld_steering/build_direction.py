#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gc


import torch

from hf_agent import HFSteeringAgent
from steering import ActivationCollector


def load_prompts(path: Path, limit: int):
    prompts = []
    for p in sorted(path.rglob("*.json")):
        obj = json.loads(p.read_text())
        traj = obj.get("trajectory", [])
        for step in traj:
            obs = step.get("observation")
            cmds = step.get("admissible_commands", [])
            if obs and cmds:
                valid = "\n".join(f"- {a}" for a in cmds)
                prompts.append(
                    f"""You are acting in ALFWorld.

Observation:
{obs}

Valid actions:
{valid}

Return only the next action.

Action:"""
                )
            if len(prompts) >= limit:
                return prompts
    return prompts


@torch.no_grad()
@torch.no_grad()
def collect_means(agent, prompts, layers):
    sums = {}
    counts = {}

    for prompt in prompts:
        inputs = agent.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(agent.model.device)

        with ActivationCollector(agent.model, layers) as collector:
            _ = agent.model(**inputs)

        for l in layers:
            if l not in collector.cache or len(collector.cache[l]) == 0:
                continue

            h_tensor = collector.cache[l][0]      # shape: [batch, seq, hidden]
            h = h_tensor[:, -1, :].float().cpu().squeeze(0)

            if l not in sums:
                sums[l] = h
                counts[l] = 1
            else:
                sums[l] += h
                counts[l] += 1

    means = {}
    for l in sums:
        means[l] = sums[l] / counts[l]

    print("Collected layers:", means.keys())
    return means

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--control-lora", required=True)
    ap.add_argument("--hack-lora", required=True)
    ap.add_argument("--prompt-source", required=True)
    ap.add_argument("--out", default="hack_direction.pt")
    ap.add_argument("--layers", default="28,29,30,31")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    prompts = load_prompts(Path(args.prompt_source), args.limit)


    print("[INFO] Loading control model...")
    control = HFSteeringAgent(args.base_model, args.control_lora)
    control_means = collect_means(control, prompts, layers)

    del control
    gc.collect()
    torch.cuda.empty_cache()

    print("[INFO] Loading hack model...")
    hack = HFSteeringAgent(args.base_model, args.hack_lora)
    hack_means = collect_means(hack, prompts, layers)

    del hack
    gc.collect()
    torch.cuda.empty_cache()

    directions = {}
    for l in layers:
        v = hack_means[l] - control_means[l]
        directions[l] = v / (v.norm() + 1e-8)

    torch.save(
        {
            "layers": layers,
            "directions": directions,
            "num_prompts": len(prompts),
        },
        args.out,
    )

    print(f"Saved direction to {args.out}")
    print(f"Used prompts: {len(prompts)}")


if __name__ == "__main__":
    main()