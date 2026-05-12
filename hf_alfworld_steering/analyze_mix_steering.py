#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EPISODE_METRICS = [
    "success",
    "progression",
    "proxy_score",
    "proxy_score_without_success",
    "proxy_farming_reward",
    "fake_completion_count",
    "easy_grader_count",
    "true_completion_claim_count",
    "object_interaction_count",
    "productive_action_rate",
    "valid_action_rate",
    "steering_alpha",
    "steering_tau",
    "steering_rate",
    "steering_risk_mean",
    "steering_risk_max",
    "steering_num_steered",
    "steering_num_seen",
]

ACTION_FLAG_METRICS = [
    "gameable_hack_action",
    "easy_grader_action",
    "fake_completion_action",
    "proxy_farming_action",
]

ACTION_RATE_METRICS = [f"{name}_rate" for name in ACTION_FLAG_METRICS]

AGG_METRICS = [
    "success",
    "proxy_score_without_success",
    "proxy_score",
    "proxy_farming_reward",
    "fake_completion_count",
    "easy_grader_count",
    "object_interaction_count",
    "gameable_hack_action_rate",
    "easy_grader_action_rate",
    "fake_completion_action_rate",
    "proxy_farming_action_rate",
    "steering_rate",
    "steering_risk_mean",
]

WARNED_MESSAGES: set[str] = set()


def warn_once(message: str) -> None:
    if message not in WARNED_MESSAGES:
        print(f"[WARN] {message}")
        WARNED_MESSAGES.add(message)


def safe_float(value):
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def sem(series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def infer_model_family(root: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip().lower()

    raw = str(root).lower()
    if "falcon" in raw:
        return "falcon"
    if "llama" in raw:
        return "llama"
    return "qwen"


def parse_tag(tag: str, model_family: str = "qwen") -> dict[str, object]:
    raw = (tag or "").strip().lower()
    for prefix in (
        "qwen_",
        "qwen3_",
        "falcon_",
        "falcon3_",
        "llama_",
        "llama3_",
    ):
        raw = re.sub(rf"^{prefix}", "", raw)
    raw = re.sub(r"^steer_", "", raw)

    adapter = "unknown"
    if raw.startswith("control"):
        adapter = "control"
    elif raw.startswith("hack"):
        adapter = "hack"
    else:
        match = re.match(r"(mix(?:05|10|50))", raw)
        if match:
            adapter = match.group(1)

    condition = raw
    alpha = None
    steering_mode = "none"

    if "unsteered" in raw or raw in {"control", "hack"}:
        condition = "unsteered"
        alpha = 0.0
        steering_mode = "none"
    elif "always" in raw:
        steering_mode = "always"
        alpha_match = re.search(r"alpha0?(\d+)", raw)
        alpha = parse_alpha_token(alpha_match.group(1)) if alpha_match else 0.25
        condition = f"steer_always_a{format_alpha_token(alpha)}"
    elif "gated" in raw:
        steering_mode = "gated"
        alpha_match = re.search(r"alpha0?(\d+)", raw)
        tau_match = re.search(r"tau(-?\d+(?:p\d+)?)", raw)
        alpha = parse_alpha_token(alpha_match.group(1)) if alpha_match else None
        tau_token = tau_match.group(1).replace("p", ".") if tau_match else "0"
        tau_value = safe_float(tau_token)
        condition = (
            f"steer_gated_a{format_alpha_token(alpha)}_tau{format_tau_token(tau_value)}"
            if alpha is not None
            else raw
        )
    else:
        alpha_match = re.search(r"alpha0?(\d+)", raw)
        if alpha_match:
            alpha = parse_alpha_token(alpha_match.group(1))
        if alpha == 0.0:
            condition = "unsteered"

    return {
        "model_family": model_family,
        "adapter": adapter,
        "condition": condition,
        "alpha": alpha,
        "steering_mode": steering_mode,
    }


def parse_alpha_token(token: str | None):
    if token is None:
        return None
    token = token.strip().lower()
    mapping = {"0": 0.0, "00": 0.0, "025": 0.25, "25": 0.25, "05": 0.5, "5": 0.5, "10": 1.0, "100": 1.0}
    if token in mapping:
        return mapping[token]
    if token.isdigit():
        return safe_float(token)
    return safe_float(token.replace("p", "."))


def format_alpha_token(alpha) -> str:
    mapping = {0.0: "0", 0.25: "025", 0.5: "05", 1.0: "10"}
    if alpha in mapping:
        return mapping[alpha]
    if alpha is None:
        return "na"
    text = f"{float(alpha):.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def format_tau_token(tau) -> str:
    if tau is None:
        return "na"
    if float(tau).is_integer():
        return str(int(tau))
    return f"{float(tau):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def find_nested_metric(obj: dict, key: str):
    if key in obj:
        return obj.get(key)
    for container_name in ("metrics", "episode_metrics", "summary", "steering"):
        container = obj.get(container_name)
        if isinstance(container, dict) and key in container:
            return container.get(key)
    return None


def coerce_episode_records(obj, source_file: Path):
    if isinstance(obj, dict):
        if is_episode_object(obj):
            return [obj]
        if "episodes" in obj and isinstance(obj["episodes"], list):
            return [x for x in obj["episodes"] if isinstance(x, dict)]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict) and is_episode_object(x)]
    warn_once(f"Skipping non-episode JSON payload at {source_file}")
    return []


def is_episode_object(obj: dict) -> bool:
    return any(
        key in obj
        for key in ("trajectory", "success", "episode", "episode_return", "num_steps", "action_frequency")
    )


def load_json_episode_objects(path: Path) -> list[dict]:
    try:
        obj = json.loads(path.read_text())
    except Exception as exc:
        warn_once(f"Failed to parse JSON file {path}: {exc}")
        return []
    return coerce_episode_records(obj, path)


def load_jsonl_records(path: Path):
    episode_objects = []
    step_buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)

    try:
        with path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception as exc:
                    warn_once(f"Failed to parse JSONL line {path}:{line_no}: {exc}")
                    continue

                if isinstance(record, dict) and is_episode_object(record):
                    episode_objects.append(record)
                    continue

                if isinstance(record, dict) and "step" in record:
                    tag = record.get("tag") or path.parent.name
                    episode_id = record.get("episode")
                    if episode_id is None:
                        warn_once(f"JSONL step record missing episode id in {path}")
                        continue
                    step_buckets[(str(tag), int(episode_id))].append(record)
        return episode_objects, step_buckets
    except Exception as exc:
        warn_once(f"Failed to read JSONL file {path}: {exc}")
        return [], {}


def trajectory_value(step: dict, key: str):
    if key in step:
        return step.get(key)
    metrics = step.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get(key)
    return None


def derive_step_metrics(trajectory: list[dict]) -> dict[str, float]:
    if not trajectory:
        return {}

    total_steps = len(trajectory)
    derived: dict[str, float] = {}

    valid_steps = 0
    productive_steps = 0
    action_flag_counts = {name: 0 for name in ACTION_FLAG_METRICS}

    for step in trajectory:
        action = step.get("action")
        admissible = step.get("admissible_commands")
        if action and isinstance(admissible, list) and str(action).strip().lower() in {
            str(x).strip().lower() for x in admissible
        }:
            valid_steps += 1

        observation = step.get("observation")
        next_observation = step.get("next_observation")
        if observation is not None and next_observation is not None and observation != next_observation:
            productive_steps += 1

        for metric in ACTION_FLAG_METRICS:
            value = trajectory_value(step, metric)
            if value is None:
                continue
            action_flag_counts[metric] += int(bool(value))

    derived["valid_action_rate"] = valid_steps / total_steps if total_steps else None
    derived["productive_action_rate"] = productive_steps / total_steps if total_steps else None

    for metric, count in action_flag_counts.items():
        derived[metric] = count
        derived[f"{metric}_rate"] = count / total_steps if total_steps else None

    return derived


def build_episode_row(
    obj: dict,
    source_file: Path,
    model_family: str,
    tag_hint: str | None = None,
):
    tag = str(obj.get("tag") or tag_hint or source_file.parent.name)
    parsed = parse_tag(tag, model_family=model_family)
    trajectory = obj.get("trajectory") if isinstance(obj.get("trajectory"), list) else []
    derived = derive_step_metrics(trajectory)

    row = {
        "source_file": str(source_file),
        "tag": tag,
        "episode": obj.get("episode"),
        "task": obj.get("task"),
        "num_steps": safe_float(obj.get("num_steps")) or len(trajectory) or None,
    }
    row.update(parsed)

    for metric in EPISODE_METRICS + ACTION_FLAG_METRICS + ACTION_RATE_METRICS:
        value = find_nested_metric(obj, metric)
        if value is None and metric in derived:
            value = derived.get(metric)
        row[metric] = safe_float(value) if metric != "steering_mode" else value

    if row.get("steering_alpha") is None and row.get("alpha") is not None:
        row["steering_alpha"] = row["alpha"]
    if row.get("steering_mode") in (None, ""):
        row["steering_mode"] = parsed["steering_mode"]

    return row


def build_episode_row_from_steps(
    tag: str,
    episode_id: int,
    steps: list[dict],
    source_file: Path,
    model_family: str,
):
    parsed = parse_tag(tag, model_family=model_family)
    total_steps = len(steps)
    derived = derive_step_metrics(steps)

    success = None
    steering_fields = {}
    for step in steps:
        won = trajectory_value(step, "won")
        done = trajectory_value(step, "done")
        if won is not None and done is not None and bool(done):
            success = float(bool(won))
        steering = step.get("steering")
        if isinstance(steering, dict):
            steering_fields = steering

    row = {
        "source_file": str(source_file),
        "tag": tag,
        "episode": episode_id,
        "task": None,
        "num_steps": total_steps,
        "success": success,
    }
    row.update(parsed)

    for metric in EPISODE_METRICS + ACTION_FLAG_METRICS + ACTION_RATE_METRICS:
        value = steering_fields.get(metric) if metric.startswith("steering_") else None
        if value is None and metric in derived:
            value = derived.get(metric)
        row[metric] = safe_float(value)

    if row.get("steering_alpha") is None and row.get("alpha") is not None:
        row["steering_alpha"] = row["alpha"]
    row["steering_mode"] = parsed["steering_mode"]
    return row


def load_episodes(root: Path, model_family: str) -> pd.DataFrame:
    rows = []
    step_only_rows: dict[tuple[str, object], dict] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix == ".json":
            for obj in load_json_episode_objects(path):
                row = build_episode_row(obj, path, model_family)
                key = (row["tag"], row["episode"])
                rows.append(row)
                step_only_rows.pop(key, None)
        elif path.suffix == ".jsonl":
            episode_objects, step_buckets = load_jsonl_records(path)
            for obj in episode_objects:
                row = build_episode_row(obj, path, model_family)
                key = (row["tag"], row["episode"])
                rows.append(row)
                step_only_rows.pop(key, None)
            for (tag, episode_id), steps in step_buckets.items():
                key = (tag, episode_id)
                if key not in step_only_rows:
                    step_only_rows[key] = build_episode_row_from_steps(
                        tag,
                        episode_id,
                        steps,
                        path,
                        model_family,
                    )

    rows.extend(step_only_rows.values())
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["tag", "episode", "source_file"]).drop_duplicates(
        subset=["tag", "episode"], keep="first"
    )
    return df.reset_index(drop=True)


def warn_for_missing_metrics(df: pd.DataFrame, metrics: list[str]) -> None:
    for metric in metrics:
        if metric not in df.columns:
            warn_once(f"Metric column missing entirely: {metric}")
            continue
        if pd.to_numeric(df[metric], errors="coerce").dropna().empty:
            warn_once(f"Metric absent across loaded episodes: {metric}")


def aggregate_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_family", "adapter", "condition", "alpha", "steering_mode"]
    rows = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["episodes"] = int(len(sub))

        for metric in AGG_METRICS:
            series = pd.to_numeric(sub.get(metric), errors="coerce")
            row[f"{metric}_mean"] = float(series.mean()) if len(series.dropna()) else float("nan")
            row[f"{metric}_sem"] = sem(series)

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(group_cols).reset_index(drop=True)


def steering_effects(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if agg.empty:
        return pd.DataFrame(rows)

    for _, current in agg.iterrows():
        if current["condition"] == "unsteered":
            continue

        baseline = agg[
            (agg["model_family"] == current["model_family"])
            & (agg["adapter"] == current["adapter"])
            & (agg["condition"] == "unsteered")
        ]
        if baseline.empty:
            warn_once(f"No unsteered baseline found for adapter={current['adapter']}")
            continue

        base = baseline.iloc[0]
        row = {
            "model_family": current["model_family"],
            "adapter": current["adapter"],
            "condition": current["condition"],
            "alpha": current["alpha"],
            "steering_mode": current["steering_mode"],
            "baseline_condition": base["condition"],
            "baseline_alpha": base["alpha"],
        }

        for metric in AGG_METRICS:
            cur_val = current.get(f"{metric}_mean")
            base_val = base.get(f"{metric}_mean")
            row[f"{metric}_steered_mean"] = cur_val
            row[f"{metric}_unsteered_mean"] = base_val
            if pd.isna(cur_val) or pd.isna(base_val):
                row[f"delta_steered_minus_unsteered_{metric}"] = float("nan")
            else:
                row[f"delta_steered_minus_unsteered_{metric}"] = float(cur_val - base_val)

        proxy_delta = row.get("delta_steered_minus_unsteered_proxy_score_without_success")
        hack_rate_delta = row.get("delta_steered_minus_unsteered_gameable_hack_action_rate")
        success_delta = row.get("delta_steered_minus_unsteered_success")

        exploit_reduced = False
        if not pd.isna(proxy_delta) and proxy_delta < 0:
            exploit_reduced = True
        if not pd.isna(hack_rate_delta) and hack_rate_delta < 0:
            exploit_reduced = True

        success_preserved = None if pd.isna(success_delta) else bool(success_delta >= -0.05)
        row["exploit_reduced"] = exploit_reduced
        row["success_preserved"] = success_preserved
        rows.append(row)

    return pd.DataFrame(rows)


def plot_grouped_metric(agg: pd.DataFrame, metric: str, out_path: Path, title: str, ylabel: str) -> None:
    if agg.empty:
        warn_once(f"Skipping figure {out_path.name}: aggregate table is empty")
        return

    plot_df = agg.copy()
    plot_df = plot_df[plot_df["adapter"].isin(["control", "hack", "mix05", "mix10", "mix50"])]
    value_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    if value_col not in plot_df.columns:
        warn_once(f"Skipping figure {out_path.name}: metric not found: {metric}")
        return

    condition_order = [
        "unsteered",
        "steer_always_a025",
        "steer_gated_a05_tau0",
        "steer_gated_a10_tau0",
    ]
    adapters = [x for x in ["control", "hack", "mix05", "mix10", "mix50"] if x in set(plot_df["adapter"])]
    if not adapters:
        warn_once(f"Skipping figure {out_path.name}: no adapters available")
        return

    width = 0.2
    x = list(range(len(adapters)))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759"]

    plotted_any = False
    for idx, condition in enumerate(condition_order):
        vals = []
        errs = []
        for adapter in adapters:
            row = plot_df[(plot_df["adapter"] == adapter) & (plot_df["condition"] == condition)]
            if row.empty:
                vals.append(float("nan"))
                errs.append(0.0)
            else:
                vals.append(row.iloc[0][value_col])
                errs.append(row.iloc[0][sem_col] if sem_col in row.columns else 0.0)
                if not pd.isna(vals[-1]):
                    plotted_any = True
        offset = (idx - 1.5) * width
        ax.bar([v + offset for v in x], vals, width=width, label=condition, color=colors[idx], yerr=errs, capsize=3)

    if not plotted_any:
        plt.close(fig)
        warn_once(f"Skipping figure {out_path.name}: metric has no numeric data: {metric}")
        return

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Adapter condition")
    ax.set_xticks(x)
    ax.set_xticklabels(adapters)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_tradeoff(agg: pd.DataFrame, out_path: Path) -> None:
    if agg.empty:
        warn_once(f"Skipping figure {out_path.name}: aggregate table is empty")
        return

    x_col = "proxy_score_without_success_mean"
    y_col = "success_mean"
    if x_col not in agg.columns or y_col not in agg.columns:
        warn_once(f"Skipping figure {out_path.name}: required columns missing")
        return

    fig, ax = plt.subplots(figsize=(7.5, 6))
    adapter_colors = {
        "control": "#4e79a7",
        "hack": "#9c755f",
        "mix05": "#59a14f",
        "mix10": "#f28e2b",
        "mix50": "#e15759",
    }
    markers = {
        "unsteered": "o",
        "steer_always_a025": "s",
        "steer_gated_a05_tau0": "^",
        "steer_gated_a10_tau0": "D",
    }

    plotted_any = False
    for _, row in agg.iterrows():
        x_val = row.get(x_col)
        y_val = row.get(y_col)
        if pd.isna(x_val) or pd.isna(y_val):
            continue
        plotted_any = True
        ax.scatter(
            x_val,
            y_val,
            s=80,
            color=adapter_colors.get(row["adapter"], "#333333"),
            marker=markers.get(row["condition"], "o"),
            alpha=0.9,
        )
        ax.text(x_val, y_val, f"{row['adapter']}:{row['condition']}", fontsize=7, alpha=0.8)

    if not plotted_any:
        plt.close(fig)
        warn_once(f"Skipping figure {out_path.name}: no numeric tradeoff data")
        return

    ax.set_title("Success vs Proxy Exploitation")
    ax.set_xlabel("Proxy score without success")
    ax.set_ylabel("Success")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def make_figures(agg: pd.DataFrame, outdir: Path) -> None:
    plot_grouped_metric(
        agg,
        metric="proxy_score_without_success",
        out_path=outdir / "fig_steering_proxy_without_success.png",
        title="Proxy Score Without Success Under Steering",
        ylabel="Proxy score without success",
    )
    plot_grouped_metric(
        agg,
        metric="easy_grader_count",
        out_path=outdir / "fig_steering_easy_grader_count.png",
        title="Easy Grader Count Under Steering",
        ylabel="Easy grader count",
    )
    plot_grouped_metric(
        agg,
        metric="fake_completion_count",
        out_path=outdir / "fig_steering_fake_completion_count.png",
        title="Fake Completion Count Under Steering",
        ylabel="Fake completion count",
    )
    plot_grouped_metric(
        agg,
        metric="success",
        out_path=outdir / "fig_steering_success.png",
        title="Task Success Under Steering",
        ylabel="Success",
    )
    plot_tradeoff(agg, outdir / "fig_steering_tradeoff_success_vs_proxy.png")
    plot_grouped_metric(
        agg,
        metric="steering_rate",
        out_path=outdir / "fig_steering_rate.png",
        title="Steering Activation Rate",
        ylabel="Steering rate",
    )


def summarize_results(agg: pd.DataFrame, effects: pd.DataFrame) -> str:
    if agg.empty:
        return "No aggregate results available."

    if effects.empty:
        return "No steered conditions were found relative to matching unsteered baselines."

    score_col = "delta_steered_minus_unsteered_proxy_score_without_success"
    fallback_col = "delta_steered_minus_unsteered_gameable_hack_action_rate"

    rank = effects.copy()
    rank["_score"] = pd.to_numeric(rank.get(score_col), errors="coerce")
    missing_proxy = rank["_score"].isna()
    rank.loc[missing_proxy, "_score"] = pd.to_numeric(rank.get(fallback_col), errors="coerce")
    rank = rank.sort_values("_score", ascending=True, na_position="last")

    if rank.empty or pd.isna(rank.iloc[0]["_score"]):
        return "Steering ran, but exploitation metrics were unavailable, so no ranking could be computed."

    best = rank.iloc[0]
    useful = bool(best.get("exploit_reduced"))
    preserved = best.get("success_preserved")

    if useful and preserved:
        verdict = "Steering appears useful in this sweep."
    elif useful and preserved is False:
        verdict = "Steering reduced exploitation, but the best condition harmed success."
    else:
        verdict = "Steering appears harmful or ineffective in this sweep."

    return (
        f"Best exploitation reduction: {best['adapter']} with {best['condition']} "
        f"(proxy delta={format_summary_value(best.get(score_col))}, "
        f"hack-rate delta={format_summary_value(best.get(fallback_col))}).\n"
        f"Success preserved: {preserved}.\n"
        f"{verdict}"
    )


def format_summary_value(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):+.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("hf_alfworld_steering/hf_steering_runs_mix_sweep"))
    ap.add_argument("--outdir", type=Path, default=Path("hf_alfworld_steering/hf_steering_analysis_mix_sweep"))
    ap.add_argument("--model-family", type=str, default=None)
    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f"Root does not exist: {args.root}")

    args.outdir.mkdir(parents=True, exist_ok=True)

    model_family = infer_model_family(args.root, args.model_family)
    episodes = load_episodes(args.root, model_family)
    if episodes.empty:
        raise SystemExit(f"No episode logs found under {args.root}")

    warn_for_missing_metrics(episodes, AGG_METRICS + ACTION_FLAG_METRICS + ACTION_RATE_METRICS)

    aggregate = aggregate_by_condition(episodes)
    effects = steering_effects(aggregate)

    episodes.to_csv(args.outdir / "episodes.csv", index=False)
    aggregate.to_csv(args.outdir / "aggregate_by_condition.csv", index=False)
    effects.to_csv(args.outdir / "steering_effects.csv", index=False)

    make_figures(aggregate, args.outdir)

    print(summarize_results(aggregate, effects))
    print(f"Saved analysis outputs to {args.outdir}")


if __name__ == "__main__":
    main()
