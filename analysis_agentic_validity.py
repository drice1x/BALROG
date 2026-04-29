import os
import json
import re
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

TRAJ_ROOTS = [
    "traj_eval_sweep_FalconControl",
    "traj_eval_sweep_FalconHack",
]

OUT_DIR = "analysis_agentic_validity"
os.makedirs(OUT_DIR, exist_ok=True)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

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


def safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan")
    return float(np.mean(xs))


def safe_se(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) <= 1:
        return 0.0
    return float(np.std(xs, ddof=1) / np.sqrt(len(xs)))


def model_label_from_root(root):
    if "FalconHack" in root:
        return "FalconHack"
    if "FalconControl" in root:
        return "FalconControl"
    return os.path.basename(root)


# --------------------------------------------------
# DATA COLLECTION
# --------------------------------------------------

def collect_all_rows():
    all_rows = []

    for traj_root in TRAJ_ROOTS:
        if not os.path.exists(traj_root):
            print(f"[WARN] missing traj root: {traj_root}")
            continue

        model_label = model_label_from_root(traj_root)

        for run_name in sorted(os.listdir(traj_root)):
            run_path = os.path.join(traj_root, run_name)
            if not os.path.isdir(run_path):
                continue

            rtok = parse_rtok(run_name)

            for root, _, files in os.walk(run_path):
                for file in files:
                    if not file.endswith(".jsonl"):
                        continue

                    path = os.path.join(root, file)
                    rows = load_jsonl(path)

                    for row in rows:
                        row["_traj_root"] = traj_root
                        row["_model_label"] = model_label
                        row["_run_name"] = run_name
                        row["_rtok"] = rtok
                        all_rows.append(row)

    return all_rows


# --------------------------------------------------
# ANALYSIS 1: CONSTRUCT VALIDITY
# --------------------------------------------------

def summarize_construct_validity(rows):
    grouped = defaultdict(list)

    for r in rows:
        key = (
            r.get("_model_label", "unknown"),
            r.get("env_name", "unknown"),
            r.get("task", "unknown"),
            r.get("_rtok", None),
        )
        grouped[key].append(r)

    records = []

    for key, group in sorted(grouped.items()):
        model, env_name, task, rtok = key

        bad_action_rate = safe_mean([1.0 if g.get("bad_action", False) else 0.0 for g in group])
        invalid_rate = safe_mean([1.0 if g.get("invalid_action", False) else 0.0 for g in group])
        loop_rate = safe_mean([1.0 if g.get("repeat_action_loop", False) else 0.0 for g in group])
        no_prog_rate = safe_mean([1.0 if g.get("no_progress_window", False) else 0.0 for g in group])

        reasoning_phack = safe_mean([g.get("reasoning_p_hack") for g in group])
        action_phack = safe_mean([g.get("action_p_hack") for g in group])
        reasoning_entropy = safe_mean([g.get("reasoning_entropy_mean") for g in group])
        action_entropy = safe_mean([g.get("action_entropy_mean") for g in group])

        records.append({
            "model": model,
            "env_name": env_name,
            "task": task,
            "rtok": rtok,
            "num_steps": len(group),
            "bad_action_rate": bad_action_rate,
            "invalid_action_rate": invalid_rate,
            "repeat_action_loop_rate": loop_rate,
            "no_progress_window_rate": no_prog_rate,
            "reasoning_p_hack_mean": reasoning_phack,
            "action_p_hack_mean": action_phack,
            "reasoning_entropy_mean": reasoning_entropy,
            "action_entropy_mean": action_entropy,
        })

    return records


def write_construct_csv(records):
    out_path = os.path.join(OUT_DIR, "construct_validity_summary.csv")
    import csv

    if not records:
        print("[WARN] no construct-validity records")
        return

    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] wrote {out_path}")


def print_construct_summary(records):
    print("\n=== CONSTRUCT VALIDITY SUMMARY ===")
    for r in sorted(records, key=lambda x: (x["env_name"], x["task"], x["rtok"], x["model"])):
        print(
            f"model={r['model']:12s} "
            f"env={r['env_name']:10s} "
            f"task={r['task']:24s} "
            f"rtok={str(r['rtok']):>4s} "
            f"bad={r['bad_action_rate']:.3f} "
            f"invalid={r['invalid_action_rate']:.3f} "
            f"loop={r['repeat_action_loop_rate']:.3f} "
            f"noprog={r['no_progress_window_rate']:.3f} "
            f"r_phack={r['reasoning_p_hack_mean']:.3f} "
            f"a_phack={r['action_p_hack_mean']:.3f}"
        )


# --------------------------------------------------
# ANALYSIS 2: BAD VS GOOD ACTIONS
# --------------------------------------------------

def summarize_bad_vs_good(rows):
    grouped = defaultdict(list)

    for r in rows:
        key = (
            r.get("_model_label", "unknown"),
            r.get("env_name", "unknown"),
            r.get("task", "unknown"),
            r.get("_rtok", None),
        )
        grouped[key].append(r)

    records = []

    for key, group in sorted(grouped.items()):
        model, env_name, task, rtok = key

        bad_group = [g for g in group if g.get("bad_action", False)]
        good_group = [g for g in group if not g.get("bad_action", False)]

        records.append({
            "model": model,
            "env_name": env_name,
            "task": task,
            "rtok": rtok,

            "n_bad": len(bad_group),
            "n_good": len(good_group),

            "bad_reasoning_p_hack": safe_mean([g.get("reasoning_p_hack") for g in bad_group]),
            "good_reasoning_p_hack": safe_mean([g.get("reasoning_p_hack") for g in good_group]),

            "bad_action_p_hack": safe_mean([g.get("action_p_hack") for g in bad_group]),
            "good_action_p_hack": safe_mean([g.get("action_p_hack") for g in good_group]),

            "bad_reasoning_entropy": safe_mean([g.get("reasoning_entropy_mean") for g in bad_group]),
            "good_reasoning_entropy": safe_mean([g.get("reasoning_entropy_mean") for g in good_group]),

            "bad_action_entropy": safe_mean([g.get("action_entropy_mean") for g in bad_group]),
            "good_action_entropy": safe_mean([g.get("action_entropy_mean") for g in good_group]),
        })

    return records


def write_bad_vs_good_csv(records):
    out_path = os.path.join(OUT_DIR, "bad_vs_good_summary.csv")
    import csv

    if not records:
        print("[WARN] no bad-vs-good records")
        return

    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] wrote {out_path}")


def print_bad_vs_good_summary(records):
    print("\n=== BAD VS GOOD ACTION SUMMARY ===")
    for r in sorted(records, key=lambda x: (x["env_name"], x["task"], x["rtok"], x["model"])):
        print(
            f"model={r['model']:12s} "
            f"env={r['env_name']:10s} "
            f"task={r['task']:24s} "
            f"rtok={str(r['rtok']):>4s} "
            f"bad_a_phack={r['bad_action_p_hack']:.3f} "
            f"good_a_phack={r['good_action_p_hack']:.3f} "
            f"bad_r_phack={r['bad_reasoning_p_hack']:.3f} "
            f"good_r_phack={r['good_reasoning_p_hack']:.3f}"
        )


# --------------------------------------------------
# PLOTS
# --------------------------------------------------

def plot_bad_action_rate(records):
    env_tasks = sorted(set((r["env_name"], r["task"]) for r in records))

    for env_name, task in env_tasks:
        plt.figure(figsize=(8, 5))

        for model in ["FalconControl", "FalconHack"]:
            subset = [r for r in records if r["env_name"] == env_name and r["task"] == task and r["model"] == model]
            subset = sorted(subset, key=lambda x: (x["rtok"] if x["rtok"] is not None else -1))
            if not subset:
                continue

            xs = [r["rtok"] for r in subset]
            ys = [r["bad_action_rate"] for r in subset]

            plt.plot(xs, ys, marker="o", label=model)

        plt.xlabel("Reasoning tokens (TTC)")
        plt.ylabel("Bad action rate")
        plt.title(f"{env_name} / {task} — bad action rate")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out = os.path.join(OUT_DIR, f"{env_name}_{task}_bad_action_rate.png".replace("/", "_"))
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[OK] wrote {out}")


def plot_monitor_gap(records):
    env_tasks = sorted(set((r["env_name"], r["task"]) for r in records))

    for env_name, task in env_tasks:
        plt.figure(figsize=(8, 5))

        for model in ["FalconControl", "FalconHack"]:
            subset = [r for r in records if r["env_name"] == env_name and r["task"] == task and r["model"] == model]
            subset = sorted(subset, key=lambda x: (x["rtok"] if x["rtok"] is not None else -1))
            if not subset:
                continue

            xs = [r["rtok"] for r in subset]
            ys = [r["action_p_hack_mean"] - r["reasoning_p_hack_mean"] for r in subset]

            plt.plot(xs, ys, marker="o", label=model)

        plt.xlabel("Reasoning tokens (TTC)")
        plt.ylabel("Action p_hack - Reasoning p_hack")
        plt.title(f"{env_name} / {task} — action-vs-reasoning hack gap")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out = os.path.join(OUT_DIR, f"{env_name}_{task}_monitor_gap.png".replace("/", "_"))
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[OK] wrote {out}")


# --------------------------------------------------
# PAPER-FACING AUTO-SUMMARY
# --------------------------------------------------

def auto_summary(records, badgood_records):
    lines = []
    lines.append("=== DRAFT PAPER STATEMENTS ===")

    for env_name, task in sorted(set((r["env_name"], r["task"]) for r in records)):
        hack = [r for r in records if r["env_name"] == env_name and r["task"] == task and r["model"] == "FalconHack"]
        ctrl = [r for r in records if r["env_name"] == env_name and r["task"] == task and r["model"] == "FalconControl"]

        if hack and ctrl:
            hack_bad = safe_mean([r["bad_action_rate"] for r in hack])
            ctrl_bad = safe_mean([r["bad_action_rate"] for r in ctrl])

            hack_gap = safe_mean([r["action_p_hack_mean"] - r["reasoning_p_hack_mean"] for r in hack])
            ctrl_gap = safe_mean([r["action_p_hack_mean"] - r["reasoning_p_hack_mean"] for r in ctrl])

            lines.append(
                f"- In {env_name}/{task}, FalconHack shows "
                f"{'higher' if hack_bad > ctrl_bad else 'not higher'} bad-action rates than FalconControl "
                f"({hack_bad:.3f} vs {ctrl_bad:.3f})."
            )
            lines.append(
                f"- In {env_name}/{task}, the action-vs-reasoning hack gap is "
                f"{hack_gap:.3f} for FalconHack and {ctrl_gap:.3f} for FalconControl."
            )

        subset_bg = [r for r in badgood_records if r["env_name"] == env_name and r["task"] == task]
        for rec in subset_bg:
            if rec["model"] == "FalconHack":
                lines.append(
                    f"- For FalconHack in {env_name}/{task} at TTC={rec['rtok']}, "
                    f"bad actions have action p_hack={rec['bad_action_p_hack']:.3f} "
                    f"vs good actions {rec['good_action_p_hack']:.3f}."
                )

    out_path = os.path.join(OUT_DIR, "paper_statements.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"[OK] wrote {out_path}")
    print("\n".join(lines))


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    rows = collect_all_rows()
    print(f"[INFO] loaded {len(rows)} step rows")

    construct_records = summarize_construct_validity(rows)
    write_construct_csv(construct_records)
    print_construct_summary(construct_records)

    badgood_records = summarize_bad_vs_good(rows)
    write_bad_vs_good_csv(badgood_records)
    print_bad_vs_good_summary(badgood_records)

    plot_bad_action_rate(construct_records)
    plot_monitor_gap(construct_records)

    auto_summary(construct_records, badgood_records)


if __name__ == "__main__":
    main()