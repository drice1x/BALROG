#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


TARGET_ORDER = [
    "bad_action",
    "gameable_hack_action",
    "easy_grader_action",
    "fake_completion_action",
]
PREDICTOR_ORDER = [
    "reasoning_entropy_mean",
    "action_entropy_mean",
    "reasoning_p_hack",
    "action_p_hack",
    "reasoning_p_hack_late_change",
    "action_p_hack_late_change",
    "reasoning_p_hack_late_slope",
    "action_p_hack_late_slope",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fnum(text: str | None) -> float | None:
    try:
        if text in (None, ""):
            return None
        return float(text)
    except Exception:
        return None


def sort_key_model(text: str) -> tuple[str, str]:
    family = "Other"
    for prefix in ("Qwen", "Llama", "Falcon"):
        if str(text).startswith(prefix):
            family = prefix
            break
    return (family, str(text))


def make_wide_rows(pred_rows: list[dict[str, str]], step_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    pred_map: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in pred_rows:
        if row.get("method") != "react_ttc_monitored":
            continue
        if row.get("target") not in TARGET_ORDER:
            continue
        key = (row["model"], row["ttc"], row["target"])
        pred_map[key][row["predictor"]] = row

    step_map: dict[tuple[str, str], dict[str, str]] = {}
    for row in step_rows:
        if row.get("method") != "react_ttc_monitored":
            continue
        step_map[(row["model"], row["ttc"])] = row

    out: list[dict[str, object]] = []
    for (model, ttc, target), predictors in pred_map.items():
        step = step_map.get((model, ttc), {})
        row: dict[str, object] = {
            "model": model,
            "adapter": step.get("adapter", ""),
            "ttc": ttc,
            "target": target,
            "target_rate": fnum(step.get(f"{target}_mean")),
            "bad_action_rate": fnum(step.get("bad_action_mean")),
            "gameable_hack_rate": fnum(step.get("gameable_hack_action_mean")),
            "easy_grader_rate": fnum(step.get("easy_grader_action_mean")),
            "fake_completion_rate": fnum(step.get("fake_completion_action_mean")),
            "reasoning_entropy_mean": fnum(step.get("reasoning_entropy_mean_mean")),
            "action_entropy_mean": fnum(step.get("action_entropy_mean_mean")),
            "reasoning_p_hack_mean": fnum(step.get("reasoning_p_hack_mean")),
            "action_p_hack_mean": fnum(step.get("action_p_hack_mean")),
        }

        first = next(iter(predictors.values()))
        row["n_bad"] = fnum(first.get("n_bad"))
        row["n_good"] = fnum(first.get("n_good"))

        for predictor in PREDICTOR_ORDER:
            entry = predictors.get(predictor, {})
            row[f"delta__{predictor}"] = fnum(entry.get("delta_bad_minus_good"))
            row[f"good__{predictor}"] = fnum(entry.get("good_mean"))
            row[f"bad__{predictor}"] = fnum(entry.get("bad_mean"))

        out.append(row)

    out.sort(key=lambda r: (sort_key_model(str(r["model"])), float(r["ttc"]), TARGET_ORDER.index(str(r["target"]))))
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:+.3f}" if abs(x) < 10 else f"{x:.1f}"
    return str(x)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# ALFWorld Temporal GPT Brief",
        "",
        "This file is for giving another model enough context to reason about the temporal ALFWorld result.",
        "",
        "## How to read the table",
        "",
        "- Each row is one `model / TTC / target` condition.",
        "- `target_rate` is how often that harmful action occurs in the run.",
        "- `delta__reasoning_entropy_mean` is the mean entropy at step `t` before a harmful step `t+1`, minus before a non-harmful step `t+1`.",
        "- `delta__reasoning_p_hack` is the same idea for the monitor probability.",
        "- Positive `delta__reasoning_p_hack` means `p_hack` is higher before the harmful next action.",
        "- Negative `delta__reasoning_entropy_mean` means entropy is lower before the harmful next action.",
        "",
        "## Compact table",
        "",
        "| Model | TTC | Target | Target Rate | n_bad | Δ reasoning entropy | Δ action entropy | Δ reasoning p_hack | Δ action p_hack | Δ reasoning late Δp_hack | Δ action late Δp_hack |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["ttc"]).replace(".0", ""),
                    str(row["target"]),
                    fmt(row["target_rate"]),
                    fmt(row["n_bad"]),
                    fmt(row["delta__reasoning_entropy_mean"]),
                    fmt(row["delta__action_entropy_mean"]),
                    fmt(row["delta__reasoning_p_hack"]),
                    fmt(row["delta__action_p_hack"]),
                    fmt(row["delta__reasoning_p_hack_late_change"]),
                    fmt(row["delta__action_p_hack_late_change"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Suggested prompt context",
            "",
            "Tell the model:",
            "",
            "1. `Gameable ALFWorld` is the main intervention benchmark.",
            "2. The paper asks whether monitor state at step `t` predicts harmful action at `t+1`.",
            "3. The most relevant columns are the entropy and `p_hack` deltas before the next harmful action.",
            "4. Ask it to identify which families and mixtures show the clearest late-stage temporal structure.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, default=Path("BALROG/analysis_gameable_alfworld_temporal"))
    ap.add_argument("--outdir", type=Path, default=Path("BALROG/alfworld_temporal_gpt_brief"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    pred_rows = read_csv(args.analysis_dir / "predictive_signal_table.csv")
    step_rows = read_csv(args.analysis_dir / "step_signal_aggregate.csv")
    rows = make_wide_rows(pred_rows, step_rows)
    write_csv(rows, args.outdir / "alfworld_temporal_brief.csv")
    write_markdown(rows, args.outdir / "alfworld_temporal_brief.md")
    print(f"[DONE] Wrote GPT brief to {args.outdir}")


if __name__ == "__main__":
    main()
