#!/usr/bin/env python3
from __future__ import annotations

import argparse
import torch

from hf_agent import HFSteeringAgent
from alfworld_eval import evaluate_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--alfworld-config", required=True)
    ap.add_argument("--task", default="pick_and_place_simple")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--reasoning-tokens", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--direction-path", default=None)
    ap.add_argument("--out-dir", default="hf_steering_runs")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    directions = {}
    if args.direction_path and args.alpha != 0.0:
        obj = torch.load(args.direction_path, map_location="cpu")
        directions = obj["directions"]

    agent = HFSteeringAgent(
        base_model=args.base_model,
        lora_path=args.lora,
        directions=directions,
        alpha=args.alpha,
        reasoning_tokens=args.reasoning_tokens,
    )

    evaluate_agent(
        agent=agent,
        config_file=args.alfworld_config,
        task=args.task,
        episodes=args.episodes,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()