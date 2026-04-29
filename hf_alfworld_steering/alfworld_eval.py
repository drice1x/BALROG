from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import alfworld
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment import get_environment


def make_alfworld_env(config_file: str):
    import sys
    old_argv = sys.argv[:]
    sys.argv = [old_argv[0], config_file]
    try:
        cfg = generic.load_config()
    finally:
        sys.argv = old_argv

    cfg["env"]["type"] = "AlfredTWEnv"
    cfg["dataset"]["eval_ood_data"] = False

    env = get_environment(cfg["env"]["type"])(cfg, train_eval="eval_in_distribution")
    return env.init_env(batch_size=1)


def extract_admissible(info):
    cmds = info.get("admissible_commands", [])
    if isinstance(cmds, list) and cmds and isinstance(cmds[0], list):
        cmds = cmds[0]
    return [str(c).strip().lower() for c in cmds]


def format_obs(text_obs, info, task):
    admissible = extract_admissible(info)
    return {
        "text": {
            "long_term_context": (
                f"{text_obs}\n\nAdmissible commands:\n"
                + "\n".join(f"- {a}" for a in admissible)
            )
        },
        "mission": task,
        "admissible_commands": admissible,
    }


def evaluate_agent(agent, config_file: str, task: str, episodes: int, max_steps: int, out_dir: str, tag: str):
    out_dir = Path(out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for ep in range(episodes):
        env = make_alfworld_env(config_file)
        agent.reset()

        obs_raw, info = env.reset()
        text_obs = obs_raw[0] if isinstance(obs_raw, (list, tuple)) else obs_raw
        info0 = info[0] if isinstance(info, (list, tuple)) else info

        obs = format_obs(text_obs, info0, task)

        action_freq = defaultdict(int)
        trajectory = []
        total_reward = 0.0
        done = False

        for step in range(max_steps):
            action, reasoning, raw_action = agent.act(obs)

            obs_raw, scores, dones, infos = env.step([action])

            text_obs = obs_raw[0] if isinstance(obs_raw, (list, tuple)) else obs_raw
            reward = scores[0] if isinstance(scores, (list, tuple)) else scores
            done = dones[0] if isinstance(dones, (list, tuple)) else dones
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos

            total_reward += float(reward)
            action_freq[action] += 1

            next_obs = format_obs(text_obs, info0, task)

            trajectory.append(
                {
                    "step": step,
                    "action": action,
                    "raw_action": raw_action,
                    "reasoning": reasoning,
                    "reward": float(reward),
                    "done": bool(done),
                    "won": bool(info0.get("won", False)),
                    "observation": obs["text"]["long_term_context"],
                    "next_observation": next_obs["text"]["long_term_context"],
                    "admissible_commands": obs["admissible_commands"],
                }
            )

            obs = next_obs

            if done:
                break

        summary = {
            "episode": ep,
            "tag": tag,
            "task": task,
            "success": float(bool(info0.get("won", False))),
            "episode_return": total_reward,
            "num_steps": len(trajectory),
            "action_frequency": dict(action_freq),
            "trajectory": trajectory,
        }

        summaries.append(summary)

        with open(out_dir / f"episode_{ep:03d}.json", "w") as f:
            json.dump(summary, f, indent=2)

        env.close()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    return summaries