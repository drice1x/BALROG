#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from labels import enrich_step_labels, episode_label_summary
from monitoring_client import MonitoringClient
from react_webshop_agent import ReactWebShopAgent
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
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-products", type=int, default=1000)
    ap.add_argument("--reward-threshold", type=float, default=0.5)
    ap.add_argument("--min-inspect-steps", type=int, default=2)
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--task-ids-file")
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

    client = MonitoringClient(args.base_url, args.model_id, api_key=args.api_key)
    agent = ReactWebShopAgent(client=client, max_tokens=args.max_tokens, temperature=args.temperature)
    env = ReactWebShopEnv(args.webshop_root, num_products=args.num_products)

    step_log = steps_dir / "steps.jsonl"
    if step_log.exists():
        step_log.unlink()

    summaries = []
    aggregate = defaultdict(float)
    for episode_idx in range(args.episodes):
        current_task_id = task_ids[episode_idx] if task_ids is not None else None
        observation, info = env.reset(task_id=current_task_id, seed=episode_idx)
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
                "entropy_mean": step.trace.entropy_mean,
                "entropy_max": step.trace.entropy_max,
                "token_entropy": step.trace.token_entropy,
                "p_hack": step.trace.p_hack,
                "p_hack_trajectory": step.trace.p_hack_trajectory,
                "prompt_monitor_prob_so_far": step.trace.prompt_monitor_prob_so_far,
                "prompt_monitor_prob_trajectory": step.trace.prompt_monitor_prob_trajectory,
            }
            trajectory.append(row)
            observation = next_observation
            if done:
                break

        enrich_step_labels(trajectory, reward_threshold=args.reward_threshold, min_inspect_steps=args.min_inspect_steps)
        for row in trajectory:
            append_jsonl(step_log, row)

        summary = {
            "episode_id": episode_idx,
            "task_id": current_task_id,
            "model": args.model_id,
            "instruction": trajectory[0]["instruction"] if trajectory else observation.get("mission", ""),
            "episode_return": total_reward,
            "num_steps": len(trajectory),
            "action_frequency": dict(action_frequency),
            "trajectory": trajectory,
        }
        summary.update(episode_label_summary(trajectory))
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                aggregate[key] += float(value)
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
    }
    with (outdir / "pilot_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(pilot_metrics, handle, indent=2)

    print(json.dumps(pilot_metrics, indent=2))

    env.close()


if __name__ == "__main__":
    main()
