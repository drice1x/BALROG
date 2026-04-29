import os
import json
import re
from collections import defaultdict

TRAJ_ROOT = "traj_eval_sweep"

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def parse_model_from_row(row):
    return row.get("action_raw_response", {}).get("model") or row.get("model") or "unknown"

def parse_rtok(run_name):
    m = re.search(r"rtok(\d+)", run_name)
    return int(m.group(1)) if m else None

def summarize_run(rows):
    n = len(rows)
    if n == 0:
        return None

    def mean_bool(key):
        vals = [1 if r.get(key, False) else 0 for r in rows]
        return sum(vals) / len(vals)

    return {
        "num_steps": n,
        "bad_action_rate": mean_bool("bad_action"),
        "invalid_action_rate": mean_bool("invalid_action"),
        "repeat_action_loop_rate": mean_bool("repeat_action_loop"),
        "no_progress_window_rate": mean_bool("no_progress_window"),
    }

def main():
    results = []

    for run in sorted(os.listdir(TRAJ_ROOT)):
        run_path = os.path.join(TRAJ_ROOT, run)
        if not os.path.isdir(run_path):
            continue

        rtok = parse_rtok(run)

        for root, _, files in os.walk(run_path):
            for file in files:
                if not file.endswith(".jsonl"):
                    continue

                path = os.path.join(root, file)
                rows = load_jsonl(path)
                if not rows:
                    continue

                env_name = rows[0].get("env_name", "unknown")
                task = rows[0].get("task", "unknown")
                model = parse_model_from_row(rows[0])

                summary = summarize_run(rows)
                if summary is None:
                    continue

                results.append({
                    "run": run,
                    "env_name": env_name,
                    "task": task,
                    "model": model,
                    "rtok": rtok,
                    **summary,
                })

    # aggregate by model / env / rtok
    grouped = defaultdict(list)
    for r in results:
        key = (r["model"], r["env_name"], r["task"], r["rtok"])
        grouped[key].append(r)

    print("\n=== Construct validity summary ===")
    for key in sorted(grouped.keys()):
        model, env_name, task, rtok = key
        group = grouped[key]

        def avg(field):
            return sum(x[field] for x in group) / len(group)

        print(
            f"model={model:12s} env={env_name:10s} task={task:20s} rtok={str(rtok):>4s} "
            f"bad={avg('bad_action_rate'):.3f} "
            f"invalid={avg('invalid_action_rate'):.3f} "
            f"loop={avg('repeat_action_loop_rate'):.3f} "
            f"noprog={avg('no_progress_window_rate'):.3f} "
            f"episodes={len(group)}"
        )

if __name__ == "__main__":
    main()