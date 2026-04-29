#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


def sem(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=1) / math.sqrt(len(s)))


def parse_method_and_ttc(run_dir_name: str) -> tuple[str, int | None]:
    if run_dir_name == "react_baseline":
        return "react_baseline", None

    m = re.match(r"ttc_rtok(\d+)", run_dir_name)
    if m:
        return "react_ttc_monitored", int(m.group(1))

    return run_dir_name, None
    
def load_episode_jsons(traj_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(list(traj_root.rglob("*_run_*.json")))

    for path in paths:
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        obj["_source_file"] = str(path)

        rel = path.relative_to(traj_root).parts
        # rel example:
        # ttc_rtok64/2026-.../alfworld/pick_and_place_simple/pick_and_place_simple_run_00.json
        obj["_run_dir"] = rel[0] if len(rel) > 0 else path.parent.name

        rows.append(obj)

    return rows
def load_episode_jsons1(traj_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(list(traj_root.rglob("*.json")) + list(traj_root.rglob("*.jsonl")))

    for path in paths:
        if path.name.endswith("summary.json"):
            continue

        # Standard JSON file
        if path.suffix == ".json":
            try:
                obj = json.loads(path.read_text())
            except Exception:
                continue
            if isinstance(obj, dict):
                obj["_source_file"] = str(path)
                parts = path.relative_to(traj_root).parts
                obj["_run_dir"] = parts[0] if len(parts) > 0 else path.parent.name
                rows.append(obj)
            continue

        # JSONL file: keep the LAST dict-like line as the episode summary
        if path.suffix == ".jsonl":
            last_obj = None
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(obj, dict):
                            last_obj = obj
            except Exception:
                continue

            if last_obj is not None:
                last_obj["_source_file"] = str(path)
                parts = path.relative_to(traj_root).parts
                last_obj["_run_dir"] = parts[0] if len(parts) > 0 else path.parent.name
                rows.append(last_obj)

    return rows

def load_episode_jsonsO1(traj_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(list(traj_root.rglob("*.json")) + list(traj_root.rglob("*.jsonl")))
    for path in paths:
        if path.name.endswith("summary.json"):
            continue
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        obj["_source_file"] = str(path)
        rows.append(obj)
    return rows

def load_episode_jsonsOLD(traj_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(traj_root.rglob("*.json")):
        if path.name.endswith("summary.json"):
            continue
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        obj["_source_file"] = str(path)
        obj["_run_dir"] = path.parts[1] if len(path.parts) > 1 else path.parent.name
        rows.append(obj)
    return rows


def flatten_episode_row(obj: dict[str, Any]) -> dict[str, Any]:
    run_dir = obj.get("_run_dir", "unknown")
    method, ttc = parse_method_and_ttc(run_dir)

    out: dict[str, Any] = {
        "run_dir": run_dir,
        "method": method,
        "ttc": ttc,
        "task": obj.get("task"),
        "seed": obj.get("seed"),
        "success": obj.get("success"),
        "progression": obj.get("progression"),
        "episode_return": obj.get("episode_return"),
        "num_steps": obj.get("num_steps"),
        "input_tokens": obj.get("input_tokens"),
        "output_tokens": obj.get("output_tokens"),
        "done": obj.get("done"),
        "alfworld_won": obj.get("alfworld_won"),
        "alfworld_score": obj.get("alfworld_score"),
        "failed_candidates_len": len(obj.get("failed_candidates", [])),
        "dominant_action": None,
        "dominant_action_count": 0,
        "unique_actions": 0,
        "source_file": obj.get("_source_file"),
    }

    action_freq = obj.get("action_frequency", {})
    if isinstance(action_freq, dict) and action_freq:
        dominant_action, dominant_count = max(action_freq.items(), key=lambda kv: kv[1])
        out["dominant_action"] = dominant_action
        out["dominant_action_count"] = dominant_count
        out["unique_actions"] = len(action_freq)

    alf_metrics = obj.get("alfworld_metrics", {})
    if isinstance(alf_metrics, dict):
        for k, v in alf_metrics.items():
            out[f"alf_{k}"] = v

    return out


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = [
        "success",
        "progression",
        "episode_return",
        "num_steps",
        "input_tokens",
        "output_tokens",
        "failed_candidates_len",
        "dominant_action_count",
        "unique_actions",
        "alf_repetition_rate",
        "alf_state_action_loop_rate",
        "alf_examine_loop_rate",
        "alf_object_churn_rate",
        "alf_no_progress_window_rate",
        "alf_ineffective_action_rate",
        "alf_reward_irrelevant_action_rate",
        "alf_productive_action_rate",
        "alf_dominant_action_fraction",
        "alf_unique_action_ratio",
        "alf_first_productive_step",
        "alf_avg_reward",
        "alf_total_steps",
    ]

    existing_metrics = [m for m in metrics if m in df.columns]

    grouped = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row["episodes"] = len(sub)

        for m in existing_metrics:
            s = pd.to_numeric(sub[m], errors="coerce")
            row[f"{m}_mean"] = float(s.mean()) if len(s.dropna()) else float("nan")
            row[f"{m}_sem"] = sem(s)

        grouped.append(row)

    out = pd.DataFrame(grouped)
    sort_cols = [c for c in ["task", "method", "ttc"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def make_paper_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "task",
        "method",
        "ttc",
        "episodes",
        "success_mean",
        "progression_mean",
        "alf_repetition_rate_mean",
        "alf_examine_loop_rate_mean",
        "alf_no_progress_window_rate_mean",
        "alf_reward_irrelevant_action_rate_mean",
        "alf_productive_action_rate_mean",
        "input_tokens_mean",
        "output_tokens_mean",
    ]
    cols = [c for c in cols if c in df.columns]
    table = df[cols].copy()

    for c in table.columns:
        if c.endswith("_mean"):
            table[c] = table[c].round(4)
    return table


def print_key_findings(df: pd.DataFrame) -> None:
    print("\n=== KEY FINDINGS ===")

    if "method" not in df.columns:
        return

    baseline = df[df["method"] == "react_baseline"].copy()
    ttc = df[df["method"] == "react_ttc_monitored"].copy()

    if not baseline.empty:
        print("\nReAct baseline:")
        cols = [c for c in [
            "task", "success_mean", "progression_mean",
            "alf_repetition_rate_mean", "alf_no_progress_window_rate_mean",
            "alf_reward_irrelevant_action_rate_mean", "episodes"
        ] if c in baseline.columns]
        print(baseline[cols].to_string(index=False))

    if not ttc.empty and "ttc" in ttc.columns:
        print("\nTTC sweep:")
        cols = [c for c in [
            "task", "ttc", "success_mean", "progression_mean",
            "alf_repetition_rate_mean", "alf_examine_loop_rate_mean",
            "alf_no_progress_window_rate_mean", "alf_reward_irrelevant_action_rate_mean",
            "alf_productive_action_rate_mean", "episodes"
        ] if c in ttc.columns]
        print(ttc[cols].to_string(index=False))

        if "alf_repetition_rate_mean" in ttc.columns:
            best_rep = ttc.sort_values("alf_repetition_rate_mean").iloc[0]
            print(
                f"\nLowest repetition-rate TTC: task={best_rep.get('task')} "
                f"ttc={best_rep.get('ttc')} "
                f"rep={best_rep.get('alf_repetition_rate_mean'):.4f}"
            )

        if "success_mean" in ttc.columns:
            best_succ = ttc.sort_values("success_mean", ascending=False).iloc[0]
            print(
                f"Best success TTC: task={best_succ.get('task')} "
                f"ttc={best_succ.get('ttc')} "
                f"success={best_succ.get('success_mean'):.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj-root",
        type=Path,
        default=Path("traj_paper_eval"),
        help="Root directory containing trajectory JSON files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_react_ttc_alfworld"),
        help="Directory for CSV outputs.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    episode_objs = load_episode_jsons(args.traj_root)
    if not episode_objs:
        raise SystemExit(f"No episode JSON files found under {args.traj_root}")

    episode_rows = [flatten_episode_row(obj) for obj in episode_objs]
    episodes_df = pd.DataFrame(episode_rows)

    episodes_csv = args.outdir / "episodes_flat.csv"
    episodes_df.to_csv(episodes_csv, index=False)

    by_run = aggregate(episodes_df, ["run_dir", "task", "method", "ttc"])
    by_run_csv = args.outdir / "aggregate_by_run.csv"
    by_run.to_csv(by_run_csv, index=False)

    by_method = aggregate(episodes_df, ["task", "method", "ttc"])
    by_method_csv = args.outdir / "aggregate_by_method.csv"
    by_method.to_csv(by_method_csv, index=False)

    paper_table = make_paper_table(by_method)
    paper_table_csv = args.outdir / "paper_table.csv"
    paper_table.to_csv(paper_table_csv, index=False)

    print(f"Saved episode-level table: {episodes_csv}")
    print(f"Saved run-level aggregate: {by_run_csv}")
    print(f"Saved method-level aggregate: {by_method_csv}")
    print(f"Saved paper table:        {paper_table_csv}")

    print("\n=== PAPER TABLE ===")
    print(paper_table.to_string(index=False))

    print_key_findings(by_method)


if __name__ == "__main__":
    main()