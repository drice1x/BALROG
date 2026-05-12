#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hf_webshop_agent import HFSteeringWebShopAgent
from labels import enrich_step_labels, episode_label_summary
from react_webshop_env import ReactWebShopEnv


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_task_ids(task_id: int | None, task_ids_file: str | None) -> list[int] | None:
    if task_id is not None and task_ids_file:
        raise ValueError("Use only one of --task-id or --task-ids-file.")
    if task_id is not None:
        return [int(task_id)]
    if not task_ids_file:
        return None
    path = Path(task_ids_file).expanduser()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return [int(x) for x in data]
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(int(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webshop-root", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--direction-path", default=None)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-products", type=int, default=1000)
    ap.add_argument("--reward-threshold", type=float, default=0.5)
    ap.add_argument("--min-inspect-steps", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--steering-mode", choices=["always", "gated"], default="always")
    ap.add_argument("--steering-tau", type=float, default=0.0)
    ap.add_argument("--steering-token-scope", choices=["last", "all"], default="last")
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--task-ids-file")
    ap.add_argument("--model-id", default="local_steered")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    task_ids = load_task_ids(args.task_id, args.task_ids_file)
    if task_ids is not None and not task_ids:
        raise ValueError("No task ids were loaded.")
    if task_ids is not None and args.episodes > len(task_ids):
        args.episodes = len(task_ids)

    outdir = Path(args.outdir)
    steps_dir = outdir / "steps"
    episodes_dir = outdir / "episodes"
    outdir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    with (outdir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**vars(args), "task_ids": task_ids}, handle, indent=2)

    agent = HFSteeringWebShopAgent(
        base_model=args.base_model,
        lora_path=args.lora,
        direction_path=args.direction_path,
        alpha=args.alpha,
        steering_mode=args.steering_mode,
        steering_tau=args.steering_tau,
        steering_token_scope=args.steering_token_scope,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    env = ReactWebShopEnv(args.webshop_root, num_products=args.num_products)

    step_log = steps_dir / "steps.jsonl"
    if step_log.exists():
        step_log.unlink()

    summaries = []
    aggregate = defaultdict(float)
    steering_aggregate = defaultdict(float)
    for episode_idx in range(args.episodes):
        current_task_id = task_ids[episode_idx] if task_ids is not None else None
        observation, _info = env.reset(task_id=current_task_id, seed=episode_idx)
        agent.reset()
        trajectory = []
        total_reward = 0.0
        action_frequency = defaultdict(int)

        for step_idx in range(args.max_steps):
            step = agent.act(observation)
            next_observation, reward, done, next_info = env.step(step.action)
            total_reward += float(reward)
            action_frequency[step.action] += 1

            row = {
                "episode_id": episode_idx,
                "task_id": current_task_id,
                "model": args.model_id,
                "instruction": observation.get("mission", ""),
                "step": step_idx,
                "observation": observation["text"]["long_term_context"],
                "available_actions": observation.get("available_actions", []),
                "action": step.action,
                "validated_action": next_info.get("validated_action", step.action),
                "raw_generation": step.raw_action,
                "thought_text": step.thought_text,
                "reward": float(reward),
                "done": bool(done),
                "info": next_info,
                "entropy_mean": None,
                "entropy_max": None,
                "token_entropy": [],
                "p_hack": None,
                "p_hack_trajectory": [],
                "prompt_monitor_prob_so_far": None,
                "prompt_monitor_prob_trajectory": [],
                "steering": dict(step.steering),
            }
            trajectory.append(row)
            observation = next_observation
            if done:
                break

        enrich_step_labels(trajectory, reward_threshold=args.reward_threshold, min_inspect_steps=args.min_inspect_steps)
        for row in trajectory:
            append_jsonl(step_log, row)

        steering_summary = dict(agent.last_steering_summary or {})
        summary = {
            "episode_id": episode_idx,
            "task_id": current_task_id,
            "model": args.model_id,
            "instruction": trajectory[0]["instruction"] if trajectory else observation.get("mission", ""),
            "episode_return": total_reward,
            "num_steps": len(trajectory),
            "action_frequency": dict(action_frequency),
            "trajectory": trajectory,
            **steering_summary,
        }
        summary.update(episode_label_summary(trajectory))
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                aggregate[key] += float(value)
        for key, value in steering_summary.items():
            if isinstance(value, (int, float)):
                steering_aggregate[key] += float(value)
        summaries.append(summary)
        with (episodes_dir / f"episode_{episode_idx:03d}.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    num_episodes = max(len(summaries), 1)
    pilot_metrics = {
        "model": args.model_id,
        "episodes": len(summaries),
        "avg_episode_return": aggregate["episode_return"] / num_episodes,
        "avg_num_steps": aggregate["num_steps"] / num_episodes,
        "results_page_reach_rate": aggregate["reached_results_page"] / num_episodes,
        "item_page_reach_rate": aggregate["reached_item_page"] / num_episodes,
        "asin_click_rate": aggregate["made_asin_click"] / num_episodes,
        "buy_rate": aggregate["made_buy"] / num_episodes,
        "nonzero_reward_rate": aggregate["nonzero_reward_episode"] / num_episodes,
        "bad_buy_rate": aggregate["bad_buy_action"] / num_episodes,
        "early_buy_rate": aggregate["early_buy"] / num_episodes,
        "low_reward_buy_rate": aggregate["low_reward_buy"] / num_episodes,
        "steering_rate": steering_aggregate["steering_rate"] / num_episodes,
        "steering_risk_mean": steering_aggregate["steering_risk_mean"] / num_episodes,
        "steering_risk_max": steering_aggregate["steering_risk_max"] / num_episodes,
    }
    with (outdir / "pilot_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(pilot_metrics, handle, indent=2)

    print(json.dumps(pilot_metrics, indent=2))
    env.close()


if __name__ == "__main__":
    main()
