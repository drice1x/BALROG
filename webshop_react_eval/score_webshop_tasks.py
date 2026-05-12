#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from react_webshop_agent import ReactWebShopAgent
from react_webshop_env import ReactWebShopEnv
from monitoring_client import MonitoringClient
from labels import enrich_step_labels, episode_label_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webshop-root", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--task-start", type=int, default=0)
    ap.add_argument("--task-end", type=int, default=49)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-products", type=int, default=1000)
    ap.add_argument("--reward-threshold", type=float, default=0.5)
    ap.add_argument("--min-inspect-steps", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-buy-reward", type=float, default=0.0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    client = MonitoringClient(args.base_url, args.model_id, api_key=args.api_key)
    agent = ReactWebShopAgent(client=client, max_tokens=args.max_tokens, temperature=args.temperature)
    env = ReactWebShopEnv(args.webshop_root, num_products=args.num_products)

    rows = []
    for task_id in range(args.task_start, args.task_end + 1):
        observation, _ = env.reset(task_id=task_id, seed=task_id)
        agent.reset()
        trajectory = []
        total_reward = 0.0

        for step_idx in range(args.max_steps):
            step = agent.act(observation)
            next_observation, reward, done, next_info = env.step(step.action)
            total_reward += float(reward)

            row = {
                "episode_id": 0,
                "task_id": task_id,
                "model": args.model_id,
                "instruction": observation.get("mission", ""),
                "step": step_idx,
                "observation": observation["text"]["long_term_context"],
                "available_actions": observation.get("available_actions", []),
                "action": step.action,
                "validated_action": next_info.get("validated_action", step.action),
                "reward": float(reward),
                "done": bool(done),
            }
            trajectory.append(row)
            observation = next_observation
            if done:
                break

        enrich_step_labels(trajectory, reward_threshold=args.reward_threshold, min_inspect_steps=args.min_inspect_steps)
        summary = episode_label_summary(trajectory)
        task_row = {
            "task_id": task_id,
            "instruction": trajectory[0]["instruction"] if trajectory else "",
            "episode_return": total_reward,
            "num_steps": len(trajectory),
            "reached_results_page": summary.get("reached_results_page", 0),
            "reached_item_page": summary.get("reached_item_page", 0),
            "made_asin_click": summary.get("made_asin_click", 0),
            "made_buy": summary.get("made_buy", 0),
            "nonzero_reward_episode": summary.get("nonzero_reward_episode", 0),
            "bad_buy_action": summary.get("bad_buy_action", 0),
            "early_buy": summary.get("early_buy", 0),
            "low_reward_buy": summary.get("low_reward_buy", 0),
        }
        score = (
            4.0 * task_row["nonzero_reward_episode"]
            + 2.0 * task_row["made_buy"]
            + 1.0 * task_row["reached_item_page"]
            + min(total_reward, 1.0)
        )
        task_row["score"] = score
        rows.append(task_row)

    env.close()

    rows.sort(key=lambda x: (x["score"], x["episode_return"], x["made_buy"], x["reached_item_page"]), reverse=True)

    scored_path = outdir / "task_scores.json"
    scored_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    curated_navigation = [
        row["task_id"]
        for row in rows
        if row["reached_item_page"]
    ][: args.top_k]

    curated_buy = [
        row["task_id"]
        for row in rows
        if row["made_buy"] or row["episode_return"] > args.min_buy_reward
    ][: args.top_k]

    (outdir / "curated_navigation_task_ids.txt").write_text(
        "".join(f"{task_id}\n" for task_id in curated_navigation),
        encoding="utf-8",
    )
    (outdir / "curated_buy_task_ids.txt").write_text(
        "".join(f"{task_id}\n" for task_id in curated_buy),
        encoding="utf-8",
    )

    summary = {
        "model": args.model_id,
        "task_range": [args.task_start, args.task_end],
        "num_scored": len(rows),
        "num_navigation_curated": len(curated_navigation),
        "num_buy_curated": len(curated_buy),
        "curated_navigation_task_ids": curated_navigation,
        "curated_buy_task_ids": curated_buy,
        "top_tasks": rows[: min(args.top_k, len(rows))],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
