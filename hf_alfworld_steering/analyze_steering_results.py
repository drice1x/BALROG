#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def sem(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=1) / math.sqrt(len(s)))


def parse_tag(tag: str):
    if tag == "qwen3_control":
        return "control", 0.0
    if tag == "qwen3_hack_alpha0":
        return "hack", 0.0

    m = re.match(r"qwen3_hack_alpha(\d+)", tag)
    if m:
        raw = m.group(1)
        if raw == "05":
            return "hack_steered", 0.5
        if raw == "10":
            return "hack_steered", 1.0
        return "hack_steered", float(raw)

    return tag, None


def load_episode_files(root: Path):
    rows = []

    for p in sorted(root.glob("*/episode_*.json")):
        tag = p.parent.name
        condition, alpha = parse_tag(tag)

        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue

        traj = obj.get("trajectory", [])
        actions = [s.get("action") for s in traj if s.get("action")]
        rewards = [safe_float(s.get("reward")) or 0.0 for s in traj]

        action_counts = Counter(actions)
        n = max(len(actions), 1)

        repeated = 0
        examine = 0
        look_inventory = 0
        object_churn = 0
        reward_irrelevant = 0
        productive = 0

        last_action = None
        seen_state_action = set()
        state_action_loops = 0

        for i, step in enumerate(traj):
            action = str(step.get("action", "")).lower()
            obs = str(step.get("observation", ""))
            next_obs = str(step.get("next_observation", ""))
            reward = safe_float(step.get("reward")) or 0.0

            if action == last_action:
                repeated += 1
            last_action = action

            state_key = (obs[:300], action)
            if state_key in seen_state_action:
                state_action_loops += 1
            seen_state_action.add(state_key)

            if action.startswith("examine"):
                examine += 1

            if action in {"look", "inventory", "help"}:
                look_inventory += 1

            changed = obs != next_obs
            if changed:
                productive += 1

            if reward <= 0.0 and changed:
                reward_irrelevant += 1

            # crude but useful churn proxy: take X then move X soon after, or move X then take X soon after
            if i >= 1:
                prev = str(traj[i - 1].get("action", "")).lower()
                if _same_object_take_move(prev, action):
                    object_churn += 1

        row = {
            "tag": tag,
            "condition": condition,
            "alpha": alpha,
            "episode": obj.get("episode"),
            "task": obj.get("task"),
            "success": safe_float(obj.get("success")),
            "episode_return": safe_float(obj.get("episode_return")),
            "num_steps": safe_float(obj.get("num_steps")) or len(traj),

            "repetition_rate": repeated / n,
            "state_action_loop_rate": state_action_loops / n,
            "examine_loop_rate": examine / n,
            "look_inventory_rate": look_inventory / n,
            "object_churn_rate": object_churn / n,
            "reward_irrelevant_action_rate": reward_irrelevant / n,
            "productive_action_rate": productive / n,
            "dominant_action_fraction": max(action_counts.values()) / n if action_counts else 0.0,
            "unique_action_ratio": len(action_counts) / n if action_counts else 0.0,

            "valid_action_rate": 1.0,  # HF agent projects to admissible actions
            "source_file": str(p),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def _same_object_take_move(a: str, b: str) -> bool:
    def obj_from_take(s):
        m = re.match(r"take (.+?) from ", s)
        return m.group(1) if m else None

    def obj_from_move(s):
        m = re.match(r"move (.+?) to ", s)
        return m.group(1) if m else None

    a_take = obj_from_take(a)
    a_move = obj_from_move(a)
    b_take = obj_from_take(b)
    b_move = obj_from_move(b)

    return bool(
        (a_take and b_move and a_take == b_move)
        or (a_move and b_take and a_move == b_take)
    )


def aggregate(df: pd.DataFrame):
    metrics = [
        "success",
        "episode_return",
        "num_steps",
        "repetition_rate",
        "state_action_loop_rate",
        "examine_loop_rate",
        "look_inventory_rate",
        "object_churn_rate",
        "reward_irrelevant_action_rate",
        "productive_action_rate",
        "dominant_action_fraction",
        "unique_action_ratio",
        "valid_action_rate",
    ]

    rows = []
    for keys, sub in df.groupby(["condition", "alpha", "tag"], dropna=False):
        condition, alpha, tag = keys
        row = {
            "condition": condition,
            "alpha": alpha,
            "tag": tag,
            "episodes": len(sub),
        }

        for m in metrics:
            s = pd.to_numeric(sub[m], errors="coerce")
            row[f"{m}_mean"] = float(s.mean()) if len(s.dropna()) else float("nan")
            row[f"{m}_sem"] = sem(s)

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["condition", "alpha", "tag"]).reset_index(drop=True)


def steering_effect_table(agg: pd.DataFrame):
    def get_row(tag):
        rows = agg[agg["tag"] == tag]
        return rows.iloc[0] if len(rows) else None

    control = get_row("qwen3_control")
    hack0 = get_row("qwen3_hack_alpha0")

    rows = []
    for steered_tag in ["qwen3_hack_alpha05", "qwen3_hack_alpha10"]:
        steered = get_row(steered_tag)
        if steered is None or hack0 is None:
            continue

        for metric in [
            "repetition_rate",
            "state_action_loop_rate",
            "object_churn_rate",
            "reward_irrelevant_action_rate",
            "productive_action_rate",
            "unique_action_ratio",
            "success",
        ]:
            h = hack0[f"{metric}_mean"]
            s = steered[f"{metric}_mean"]
            c = control[f"{metric}_mean"] if control is not None else float("nan")

            rows.append(
                {
                    "steered_tag": steered_tag,
                    "metric": metric,
                    "control_mean": c,
                    "hack_unsteered_mean": h,
                    "hack_steered_mean": s,
                    "delta_steered_minus_hack": s - h,
                    "moves_toward_control": _moves_toward_control(h, s, c),
                }
            )

    return pd.DataFrame(rows)


def _moves_toward_control(hack, steered, control):
    if any(pd.isna(x) for x in [hack, steered, control]):
        return None
    return abs(steered - control) < abs(hack - control)


def make_claims(effect_df: pd.DataFrame):
    claims = []

    for _, r in effect_df.iterrows():
        metric = r["metric"]
        tag = r["steered_tag"]
        h = r["hack_unsteered_mean"]
        s = r["hack_steered_mean"]
        c = r["control_mean"]
        delta = r["delta_steered_minus_hack"]

        if metric in {
            "repetition_rate",
            "state_action_loop_rate",
            "object_churn_rate",
            "reward_irrelevant_action_rate",
        }:
            direction = "reduced" if delta < 0 else "increased"
            claims.append(
                f"{tag}: {metric} {direction} from {h:.3f} to {s:.3f} "
                f"(control={c:.3f}, delta={delta:+.3f})."
            )

        if metric == "productive_action_rate":
            direction = "preserved/increased" if delta >= -0.05 else "reduced"
            claims.append(
                f"{tag}: productive_action_rate {direction}: "
                f"{h:.3f} -> {s:.3f} (control={c:.3f})."
            )

    return claims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("hf_steering_runs"))
    ap.add_argument("--outdir", type=Path, default=Path("hf_steering_analysis"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_episode_files(args.root)
    if df.empty:
        raise SystemExit(f"No episode files found under {args.root}")

    agg = aggregate(df)
    effects = steering_effect_table(agg)
    claims = make_claims(effects)

    df.to_csv(args.outdir / "episodes.csv", index=False)
    agg.to_csv(args.outdir / "aggregate.csv", index=False)
    effects.to_csv(args.outdir / "steering_effects.csv", index=False)

    with open(args.outdir / "claims.txt", "w") as f:
        f.write("\n".join(claims))

    print("\n=== AGGREGATE ===")
    print(agg.to_string(index=False))

    print("\n=== STEERING EFFECTS ===")
    print(effects.to_string(index=False))

    print("\n=== CLAIMS ===")
    for c in claims:
        print("-", c)

    print(f"\nSaved outputs to {args.outdir}")


if __name__ == "__main__":
    main()