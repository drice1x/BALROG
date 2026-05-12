#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_tag(tag: str) -> dict[str, str | float]:
    parts = tag.split("_")
    prefix = parts[0]
    if "unsteered" in tag:
        return {"prefix": prefix, "condition": "unsteered", "alpha": 0.0, "steering_mode": "none"}
    if "always" in tag:
        alpha = next((p.replace("alpha", "") for p in parts if p.startswith("alpha")), "0")
        return {"prefix": prefix, "condition": tag, "alpha": float(alpha), "steering_mode": "always"}
    if "gated" in tag:
        alpha = next((p.replace("alpha", "") for p in parts if p.startswith("alpha")), "0")
        tau = next((p.replace("tau", "") for p in parts if p.startswith("tau")), "0")
        return {"prefix": prefix, "condition": tag, "alpha": float(alpha), "tau": float(tau), "steering_mode": "gated"}
    return {"prefix": prefix, "condition": tag, "alpha": 0.0, "steering_mode": "unknown"}


def load_runs(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/pilot_metrics.json")):
        metrics = json.loads(path.read_text())
        parsed = parse_tag(path.parent.name)
        row = dict(metrics)
        row.update(parsed)
        row["run_dir"] = str(path.parent)
        rows.append(row)
    return rows


def steering_effects(rows: list[dict]) -> list[dict]:
    baselines: dict[str, dict] = {}
    for row in rows:
        if row.get("condition") == "unsteered":
            baselines[str(row["prefix"])] = row

    effects = []
    for row in rows:
        if row.get("condition") == "unsteered":
            continue
        baseline = baselines.get(str(row["prefix"]))
        if not baseline:
            continue
        effects.append(
            {
                "prefix": row["prefix"],
                "condition": row["condition"],
                "steering_mode": row["steering_mode"],
                "alpha": row.get("alpha", 0.0),
                "buy_rate": row.get("buy_rate"),
                "buy_rate_unsteered": baseline.get("buy_rate"),
                "delta_buy_rate": (row.get("buy_rate") or 0.0) - (baseline.get("buy_rate") or 0.0),
                "bad_buy_rate": row.get("bad_buy_rate"),
                "bad_buy_rate_unsteered": baseline.get("bad_buy_rate"),
                "delta_bad_buy_rate": (row.get("bad_buy_rate") or 0.0) - (baseline.get("bad_buy_rate") or 0.0),
                "low_reward_buy_rate": row.get("low_reward_buy_rate"),
                "low_reward_buy_rate_unsteered": baseline.get("low_reward_buy_rate"),
                "delta_low_reward_buy_rate": (row.get("low_reward_buy_rate") or 0.0) - (baseline.get("low_reward_buy_rate") or 0.0),
                "avg_episode_return": row.get("avg_episode_return"),
                "avg_episode_return_unsteered": baseline.get("avg_episode_return"),
                "delta_avg_episode_return": (row.get("avg_episode_return") or 0.0) - (baseline.get("avg_episode_return") or 0.0),
                "steering_rate": row.get("steering_rate"),
                "steering_risk_mean": row.get("steering_risk_mean"),
            }
        )
    return effects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_runs(runs_root)
    effects = steering_effects(rows)

    (outdir / "steering_runs.json").write_text(json.dumps(rows, indent=2))
    (outdir / "steering_effects.json").write_text(json.dumps(effects, indent=2))

    print(json.dumps(effects, indent=2))


if __name__ == "__main__":
    main()
