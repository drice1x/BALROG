from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from balrog.environments import make_env


STEP_STATS_FIELDS = [
    "success",
    "progression",
    "proxy_score",
    "last_proxy_reward",
    "proxy_score_without_success",
    "proxy_farming_reward",
    "fake_completion_count",
    "easy_grader_count",
    "true_completion_claim_count",
    "object_interaction_count",
    "last_hack_event",
    "grader",
    "alfworld_won",
    "alfworld_score",
    "strict_grader_count",
    "inspect_score_count",
]


def build_eval_config(task: str, max_steps: int, seed: int = 0):
    return OmegaConf.create(
        {
            "eval": {
                "max_steps_per_episode": max_steps,
            },
            "envs": {
                "env_kwargs": {
                    "seed": seed,
                }
            },
            "tasks": {
                "alfworld_tasks": [task],
            },
        }
    )


def collect_env_stats(env) -> dict:
    if not hasattr(env, "get_stats"):
        return {}
    stats = env.get_stats()
    return stats if isinstance(stats, dict) else {}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def evaluate_agent(agent, config_file: str, task: str, episodes: int, max_steps: int, out_dir: str, tag: str):
    del config_file  # HF steering now uses the BALROG Gameable ALFWorld wrapper directly.

    out_dir = Path(out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for ep in range(episodes):
        config = build_eval_config(task=task, max_steps=max_steps, seed=ep)
        env = make_env("alfworld", task, config)
        agent.reset()

        obs, info = env.reset(seed=ep)
        initial_stats = collect_env_stats(env)

        action_freq = defaultdict(int)
        trajectory = []
        total_reward = 0.0
        done = False

        step_jsonl_path = out_dir / f"episode_{ep:03d}.jsonl"
        if step_jsonl_path.exists():
            step_jsonl_path.unlink()

        for step in range(max_steps):
            obs_before = obs
            stats_before = collect_env_stats(env)

            action, reasoning, raw_action = agent.act(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
            action_freq[action] += 1

            stats_after = collect_env_stats(env)
            success = float(stats_after.get("success", 0.0)) if stats_after else float(bool(info.get("won", False)))

            step_row = {
                "episode": ep,
                "tag": tag,
                "task": task,
                "step": step,
                "action": action,
                "raw_action": raw_action,
                "reasoning": reasoning,
                "reward": float(reward),
                "done": done,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "success": success,
                "won": bool(stats_after.get("alfworld_won", info.get("won", False))) if isinstance(info, dict) else bool(stats_after.get("alfworld_won", False)),
                "observation": obs_before["text"]["long_term_context"],
                "next_observation": next_obs["text"]["long_term_context"],
                "admissible_commands": list(obs_before.get("admissible_commands", [])),
                "next_admissible_commands": list(next_obs.get("admissible_commands", [])),
                "steering": dict(getattr(agent, "last_steering_summary", {}) or {}),
                "info": info,
            }
            for field in STEP_STATS_FIELDS:
                step_row[field] = stats_after.get(field)
            if stats_before:
                step_row["stats_before"] = stats_before

            trajectory.append(step_row)
            append_jsonl(step_jsonl_path, step_row)

            obs = next_obs

            if done:
                break

        final_stats = collect_env_stats(env)
        success = float(final_stats.get("success", 0.0)) if final_stats else 0.0

        summary = {
            "episode": ep,
            "tag": tag,
            "task": task,
            "success": success,
            "episode_return": total_reward,
            "num_steps": len(trajectory),
            "action_frequency": dict(action_freq),
            "trajectory": trajectory,
            "initial_stats": initial_stats,
        }
        steering_summary = dict(getattr(agent, "last_steering_summary", {}) or {})
        summary.update(steering_summary)
        summary.update(final_stats)
        if steering_summary:
            summary["steering"] = steering_summary

        summaries.append(summary)

        with (out_dir / f"episode_{ep:03d}.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        env.close()

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    return summaries


# Smoke test:
# python3 hf_alfworld_steering/run_steering_eval.py --base-model "Qwen/Qwen3-8B-Base" --lora "/home/patrick/hacking_monitoringpipeline/qwen3_8b_sft_reward_hacker_control" --alfworld-config "/home/patrick/vllmPatrickMonitoring/alfworld/configs/base_config.yaml" --episodes 1 --max-steps 5 --reasoning-tokens 32 --out-dir /tmp/hf_steering_smoke_gameable --tag smoke_gameable
