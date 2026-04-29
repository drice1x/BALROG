# analyze_ttc_scaling.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re

TRAJ_ROOT = "traj_eval_sweep_Hack"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_rtok(run_name):
    m = re.search(r"rtok(\d+)", run_name)
    return int(m.group(1)) if m else None


def analyze_env(env_name):
    results = {}

    for run in os.listdir(TRAJ_ROOT):
        if env_name not in run:
            continue

        rtok = parse_rtok(run)
        if rtok is None:
            continue

        run_path = os.path.join(TRAJ_ROOT, run)

        reasoning_entropy_vals = []
        reasoning_phack_vals = []
        action_entropy_vals = []
        action_phack_vals = []

        for root, _, files in os.walk(run_path):
            for file in files:
                if not file.endswith(".jsonl"):
                    continue

                rows = load_jsonl(os.path.join(root, file))

                for row in rows:
                    if row.get("reasoning_entropy_mean") is not None:
                        reasoning_entropy_vals.append(row["reasoning_entropy_mean"])
                    if row.get("reasoning_p_hack") is not None:
                        reasoning_phack_vals.append(row["reasoning_p_hack"])
                    if row.get("action_entropy_mean") is not None:
                        action_entropy_vals.append(row["action_entropy_mean"])
                    if row.get("action_p_hack") is not None:
                        action_phack_vals.append(row["action_p_hack"])

        if reasoning_entropy_vals or reasoning_phack_vals or action_entropy_vals or action_phack_vals:
            results[rtok] = {
                "reasoning_entropy": np.mean(reasoning_entropy_vals) if reasoning_entropy_vals else np.nan,
                "reasoning_phack": np.mean(reasoning_phack_vals) if reasoning_phack_vals else np.nan,
                "action_entropy": np.mean(action_entropy_vals) if action_entropy_vals else np.nan,
                "action_phack": np.mean(action_phack_vals) if action_phack_vals else np.nan,
            }

    return results


def plot_env(env_name):
    data = analyze_env(env_name)
    if not data:
        print(f"[WARN] No data for {env_name}")
        return

    rtoks = sorted(data.keys())

    reasoning_entropy = [data[r]["reasoning_entropy"] for r in rtoks]
    reasoning_phack = [data[r]["reasoning_phack"] for r in rtoks]
    action_entropy = [data[r]["action_entropy"] for r in rtoks]
    action_phack = [data[r]["action_phack"] for r in rtoks]

    plt.figure(figsize=(9, 5))
    plt.plot(rtoks, reasoning_entropy, marker="o", label="Reasoning entropy")
    plt.plot(rtoks, reasoning_phack, marker="o", label="Reasoning p_hack")
    plt.plot(rtoks, action_entropy, marker="o", label="Action entropy")
    plt.plot(rtoks, action_phack, marker="o", label="Action p_hack")

    plt.xlabel("Reasoning tokens (TTC)")
    plt.ylabel("Mean value")
    plt.title(f"{env_name} — TTC scaling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = f"{env_name}_ttc_scaling_QwenHack.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    for env in ["textworld", "babyai", "minihack"]:
        plot_env(env)