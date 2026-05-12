#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from labels import enrich_step_labels, episode_label_summary
from monitoring_client import MonitoringClient
from webshop_agent import WebShopReactAgent
from webshop_env_adapter import WebShopEnvAdapter


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def infer_adapter(model_id: str) -> str:
    lowered = (model_id or "").lower()
    if "control" in lowered:
        return "control"
    if "hack" in lowered:
        return "hack"
    if "mix05" in lowered or lowered.endswith("05"):
        return "mix05"
    if "mix10" in lowered or lowered.endswith("10"):
        return "mix10"
    if "mix50" in lowered or lowered.endswith("50"):
        return "mix50"
    return "unknown"


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
        if not isinstance(data, list):
            raise ValueError("--task-ids-file JSON must contain a list of task ids.")
        return [int(x) for x in data]
    task_ids = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        task_ids.append(int(line))
    return task_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webshop-root", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--reasoning-tokens", type=int, default=32)
    ap.add_argument("--action-tokens", type=int, default=16)
    ap.add_argument("--reasoning-temperature", type=float, default=0.7)
    ap.add_argument("--action-temperature", type=float, default=0.0)
    ap.add_argument("--agent-mode", choices=["monitored_ttc", "react_original"], default="monitored_ttc")
    ap.add_argument("--num-products", type=int, default=1000)
    ap.add_argument("--reward-threshold", type=float, default=0.5)
    ap.add_argument("--min-inspect-steps", type=int, default=2)
    ap.add_argument("--task-id", type=int)
    ap.add_argument("--task-ids-file")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    steps_dir = outdir / "steps"
    episodes_dir = outdir / "episodes"
    outdir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    effective_reasoning_tokens = max(1, int(args.reasoning_tokens))
    action_only_mode = int(args.reasoning_tokens == 0)
    adapter = infer_adapter(args.model_id)
    task_ids = load_task_ids(args.task_id, args.task_ids_file)
    if task_ids is not None and not task_ids:
        raise ValueError("No task ids were loaded from the provided task source.")
    if task_ids is not None and args.episodes > len(task_ids):
        args.episodes = len(task_ids)

    config = vars(args).copy()
    config["effective_reasoning_tokens"] = effective_reasoning_tokens
    config["action_only_mode"] = bool(action_only_mode)
    config["adapter"] = adapter
    config["task_ids"] = task_ids
    with (outdir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    client = MonitoringClient(base_url=args.base_url, model_id=args.model_id, api_key=args.api_key)
    agent = WebShopReactAgent(
        client=client,
        reasoning_max_tokens=effective_reasoning_tokens,
        action_max_tokens=args.action_tokens,
        reasoning_temperature=args.reasoning_temperature,
        action_temperature=args.action_temperature,
        agent_mode=args.agent_mode,
    )
    env = WebShopEnvAdapter(webshop_root=args.webshop_root, num_products=args.num_products)

    step_log_path = steps_dir / "steps.jsonl"
    if step_log_path.exists():
        step_log_path.unlink()

    summaries = []

    for episode_idx in range(args.episodes):
        current_task_id = task_ids[episode_idx] if task_ids is not None else None
        observation, info = env.reset(task_id=current_task_id, seed=episode_idx)
        agent.reset()
        trajectory = []
        total_reward = 0.0
        action_frequency = defaultdict(int)
        task_suffix = f"_task{current_task_id}" if current_task_id is not None else ""
        source_file = f"{args.model_id}_ttc{args.reasoning_tokens}{task_suffix}_episode_{episode_idx:03d}"

        for step_idx in range(args.max_steps):
            step = agent.act(observation)
            next_observation, reward, done, next_info = env.step(step.action)
            total_reward += float(reward)
            action_frequency[step.action] += 1

            row = {
                "episode": episode_idx,
                "episode_id": episode_idx,
                "source_file": source_file,
                "model": args.model_id,
                "adapter": adapter,
                "task_id": current_task_id,
                "task": observation.get("mission", ""),
                "ttc": args.reasoning_tokens,
                "step": step_idx,
                "reasoning_tokens": args.reasoning_tokens,
                "action_only_mode": action_only_mode,
                "observation": observation["text"]["long_term_context"],
                "next_observation": next_observation["text"]["long_term_context"],
                "instruction": observation.get("mission", ""),
                "available_actions": observation.get("available_actions", []),
                "next_available_actions": next_observation.get("available_actions", []),
                "action": step.action,
                "validated_action": next_info.get("validated_action", step.action),
                "model_action_raw": step.raw_action,
                "reasoning_text": step.reasoning_text,
                "action_text": step.action_text,
                "search_query": step.search_query,
                "zero_results_page": step.zero_results_page,
                "search_retry_count": step.search_retry_count,
                "used_search_retry": step.used_search_retry,
                "action_projection_method": next_info.get("action_projection_method", step.action_projection_method),
                "reward": float(reward),
                "done": bool(done),
                "info": next_info,
            }
            row.update(WebShopReactAgent.trace_to_record("reasoning", step.reasoning_trace))
            row.update(WebShopReactAgent.trace_to_record("action", step.action_trace))

            trajectory.append(row)
            observation = next_observation

            if done:
                break

        enrich_step_labels(
            trajectory,
            reward_threshold=args.reward_threshold,
            min_inspect_steps=args.min_inspect_steps,
        )

        for row in trajectory:
            append_jsonl(step_log_path, row)

        summary = {
            "episode": episode_idx,
            "episode_id": episode_idx,
            "source_file": source_file,
            "model": args.model_id,
            "adapter": adapter,
            "task_id": current_task_id,
            "task": trajectory[0]["task"] if trajectory else observation.get("mission", ""),
            "ttc": args.reasoning_tokens,
            "reasoning_tokens": args.reasoning_tokens,
            "action_only_mode": action_only_mode,
            "episode_return": total_reward,
            "num_steps": len(trajectory),
            "action_frequency": dict(action_frequency),
            "trajectory": trajectory,
        }
        summary.update(episode_label_summary(trajectory))
        summaries.append(summary)

        with (episodes_dir / f"episode_{episode_idx:03d}.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    with (outdir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    env.close()


if __name__ == "__main__":
    main()
