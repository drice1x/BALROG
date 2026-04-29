# analyze_trajectories.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt

TRAJ_ROOT = "traj_eval_sweep_Hack"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_step_series(rows):
    reasoning_entropy = []
    reasoning_phack = []
    action_entropy = []
    action_phack = []

    for row in rows:
        if row.get("reasoning_entropy_mean") is not None:
            reasoning_entropy.append(row["reasoning_entropy_mean"])
        if row.get("reasoning_p_hack") is not None:
            reasoning_phack.append(row["reasoning_p_hack"])
        if row.get("action_entropy_mean") is not None:
            action_entropy.append(row["action_entropy_mean"])
        if row.get("action_p_hack") is not None:
            action_phack.append(row["action_p_hack"])

    return reasoning_entropy, reasoning_phack, action_entropy, action_phack


def pad_and_mean(sequences):
    max_len = max(len(s) for s in sequences)
    padded = np.array([
        s + [np.nan] * (max_len - len(s)) for s in sequences
        for s in sequences
    ])
    return np.nanmean(padded, axis=0)


def gather_env_sequences(env_name):
    re_seqs, rp_seqs, ae_seqs, ap_seqs = [], [], [], []

    for run in os.listdir(TRAJ_ROOT):
        if env_name not in run:
            continue

        run_path = os.path.join(TRAJ_ROOT, run)
        for root, _, files in os.walk(run_path):
            for file in files:
                if not file.endswith(".jsonl"):
                    continue
                rows = load_jsonl(os.path.join(root, file))
                re_s, rp_s, ae_s, ap_s = extract_step_series(rows)

                if re_s:
                    re_seqs.append(re_s)
                if rp_s:
                    rp_seqs.append(rp_s)
                if ae_s:
                    ae_seqs.append(ae_s)
                if ap_s:
                    ap_seqs.append(ap_s)

    return re_seqs, rp_seqs, ae_seqs, ap_seqs


def plot_env(env_name):
    re_seqs, rp_seqs, ae_seqs, ap_seqs = gather_env_sequences(env_name)

    if not re_seqs and not rp_seqs and not ae_seqs and not ap_seqs:
        print(f"[WARN] No data for {env_name}")
        return

    plt.figure(figsize=(9, 5))

    if re_seqs:
        plt.plot(pad_and_mean(re_seqs), label="Reasoning entropy", linewidth=2)
    if rp_seqs:
        plt.plot(pad_and_mean(rp_seqs), label="Reasoning p_hack", linewidth=2)
    if ae_seqs:
        plt.plot(pad_and_mean(ae_seqs), label="Action entropy", linewidth=2)
    if ap_seqs:
        plt.plot(pad_and_mean(ap_seqs), label="Action p_hack", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"{env_name} — Trajectory dynamics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = f"{env_name}_trajectory_QwenHack.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    for env in ["textworld", "babyai", "minihack"]:
        plot_env(env)