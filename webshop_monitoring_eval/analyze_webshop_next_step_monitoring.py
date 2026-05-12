#!/usr/bin/env python3
"""
This script evaluates whether reasoning-phase internal features at step t forecast
proxy-exploit-like or bad-buy actions at step t+1 in WebShop. The target is
environment-defined and independent of the monitor. The primary analysis avoids
same-step action features to reduce leakage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_GROUP_ORDER = ["base_step", "entropy_only", "mean_phack", "late_phack", "internal_all"]


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


def load_rows(runs_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(runs_root.rglob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
    return pd.DataFrame(rows)


def make_future_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    out["future_bad_buy"] = 0
    for _, sub in out.groupby("source_file", dropna=False):
        sub_sorted = sub.sort_values("step")
        next_vals = pd.to_numeric(sub_sorted[target], errors="coerce").fillna(0).astype(int).shift(-1).fillna(0).astype(int)
        out.loc[sub_sorted.index, "future_bad_buy"] = next_vals
    return out


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    def present(features):
        return [x for x in features if x in df.columns]

    return {
        "base_step": present(["step"]),
        "entropy_only": present(["reasoning_entropy_mean", "reasoning_entropy_late_change", "reasoning_entropy_late_slope"]),
        "mean_phack": present(["reasoning_p_hack"]),
        "late_phack": present(["reasoning_p_hack_late_change", "reasoning_p_hack_late_slope"]),
        "internal_all": present(
            [
                "reasoning_p_hack",
                "reasoning_p_hack_late_change",
                "reasoning_p_hack_late_slope",
                "reasoning_entropy_mean",
                "reasoning_entropy_late_change",
                "reasoning_entropy_late_slope",
            ]
        ),
    }


def aggregate_keys(aggregate_ttc: bool):
    if aggregate_ttc:
        return ["model", "adapter", "ttc_group"]
    return ["model", "adapter", "ttc"]


def recall_precision_at_5(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(y_true) == 0:
        return float("nan"), float("nan")
    k = max(1, int(math.ceil(len(y_true) * 0.05)))
    idx = np.argsort(-scores)[:k]
    positives = int(y_true.sum())
    tp = int(y_true[idx].sum())
    recall = 0.0 if positives == 0 else tp / positives
    precision = tp / len(idx)
    return float(recall), float(precision)


def evaluate(sub: pd.DataFrame, target: str, features: list[str], min_positives: int):
    required = ["source_file", target] + features
    d = sub[required].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty or len(d["source_file"].unique()) < 2:
        return None

    positives = int(pd.to_numeric(d[target], errors="coerce").fillna(0).sum())
    negatives = int(len(d) - positives)
    if positives < min_positives or negatives < 10:
        return None

    y_all = []
    score_all = []
    for holdout in sorted(d["source_file"].unique()):
        train = d[d["source_file"] != holdout]
        test = d[d["source_file"] == holdout]
        y_train = train[target].astype(int).values
        if len(np.unique(y_train)) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(train[features].values, y_train)
        scores = clf.predict_proba(test[features].values)[:, 1]
        y_all.extend(test[target].astype(int).tolist())
        score_all.extend(scores.tolist())

    if len(y_all) == 0 or len(set(y_all)) < 2:
        return None

    y_arr = np.asarray(y_all, dtype=int)
    score_arr = np.asarray(score_all, dtype=float)
    base_rate = float(y_arr.mean())
    recall5, precision5 = recall_precision_at_5(y_arr, score_arr)
    auprc = float(average_precision_score(y_arr, score_arr))
    return {
        "n": int(len(y_arr)),
        "positives": int(y_arr.sum()),
        "base_rate": base_rate,
        "auroc": float(roc_auc_score(y_arr, score_arr)),
        "auprc": auprc,
        "auprc_gain": float(auprc - base_rate),
        "recall_at_5pct": recall5,
        "precision_at_5pct": precision5,
    }


def plot_feature_group(summary: pd.DataFrame, outdir: Path, metric: str, filename: str, ylabel: str):
    if summary.empty:
        return
    plot_df = summary.groupby("feature_group", as_index=False)[metric].mean()
    plot_df["feature_group"] = pd.Categorical(plot_df["feature_group"], categories=FEATURE_GROUP_ORDER, ordered=True)
    plot_df = plot_df.sort_values("feature_group")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["feature_group"], plot_df[metric], color="#4e79a7")
    ax.axhline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.5)
    ax.set_xlabel("Feature group")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=250)
    plt.close(fig)


def plot_by_model(summary: pd.DataFrame, outdir: Path):
    if summary.empty:
        return
    pivot = summary.pivot_table(index="model", columns="feature_group", values="mean_auprc_gain", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(pivot.index))
    groups = [g for g in FEATURE_GROUP_ORDER if g in pivot.columns]
    width = 0.14 if groups else 0.2
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2"]
    for idx, group in enumerate(groups):
        ax.bar(x + (idx - (len(groups) - 1) / 2) * width, pivot[group].values, width=width, label=group, color=colors[idx % len(colors)])
    ax.axhline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean AUPRC gain")
    ax.set_title("WebShop next-step monitoring by model")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fig_webshop_by_model_auprc_gain.png", dpi=250)
    plt.close(fig)


def plot_bad_buy_rate_vs_ttc(df: pd.DataFrame, outdir: Path):
    if df.empty or "bad_buy_action" not in df.columns or "ttc" not in df.columns:
        return
    plot_df = (
        df.groupby("ttc", as_index=False)
        .agg(bad_buy_rate=("bad_buy_action", "mean"))
        .sort_values("ttc")
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_df["ttc"], plot_df["bad_buy_rate"], marker="o", color="#4e79a7")
    ax.set_xlabel("TTC / reasoning tokens")
    ax.set_ylabel("Bad buy rate")
    ax.set_title("Bad/proxy-buy rate vs TTC")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "fig_bad_buy_rate_vs_ttc.png", dpi=250)
    plt.close(fig)


def plot_metric_vs_ttc(main_df: pd.DataFrame, outdir: Path, feature_group: str, metric: str, filename: str, ylabel: str):
    if main_df.empty or "ttc" not in main_df.columns:
        return
    plot_df = main_df[main_df["feature_group"] == feature_group].copy()
    if plot_df.empty:
        return
    plot_df = plot_df.groupby("ttc", as_index=False)[metric].mean().sort_values("ttc")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_df["ttc"], plot_df[metric], marker="o", color="#59a14f")
    ax.axhline(0.0, linestyle="--", linewidth=1, color="black", alpha=0.5)
    ax.set_xlabel("TTC / reasoning tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=250)
    plt.close(fig)


def plot_signal_vs_ttc(df: pd.DataFrame, outdir: Path, candidates: list[str], filename: str, ylabel: str):
    if df.empty or "ttc" not in df.columns:
        return
    metric = next((c for c in candidates if c in df.columns), None)
    if metric is None:
        return
    plot_df = df.groupby("ttc", as_index=False)[metric].mean().sort_values("ttc")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_df["ttc"], plot_df[metric], marker="o", color="#f28e2b")
    ax.set_xlabel("TTC / reasoning tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=250)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("webshop_runs"))
    ap.add_argument("--outdir", type=Path, default=Path("webshop_analysis"))
    ap.add_argument("--min-positives", type=int, default=10)
    ap.add_argument("--aggregate-ttc", action="store_true")
    args = ap.parse_args()

    if not args.runs_root.exists():
        raise SystemExit(f"Runs root does not exist: {args.runs_root}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_rows(args.runs_root)
    if df.empty:
        raise SystemExit(f"No step logs found under {args.runs_root}")

    if "bad_buy_action" not in df.columns:
        raise SystemExit("WebShop step logs do not contain 'bad_buy_action'. Re-run evaluation with labels enabled.")

    if "adapter" not in df.columns:
        df["adapter"] = "unknown"
    if "ttc" not in df.columns:
        df["ttc"] = np.nan
    df["ttc_group"] = "ALL" if args.aggregate_ttc else df["ttc"]
    df = make_future_target(df, "bad_buy_action")
    groups = feature_groups(df)
    rows = []
    grouping = aggregate_keys(args.aggregate_ttc)
    for keys, sub in df.groupby(grouping, dropna=False):
        base = dict(zip(grouping, keys))
        for feature_group, features in groups.items():
            if not features:
                continue
            res = evaluate(sub, "future_bad_buy", features, args.min_positives)
            if res is None:
                continue
            output_ttc = "ALL" if args.aggregate_ttc else base.get("ttc", np.nan)
            rows.append(
                {
                    **base,
                    "ttc": output_ttc,
                    "target": "future_bad_buy",
                    "feature_group": feature_group,
                    **res,
                }
            )

    main_df = pd.DataFrame(rows)
    if main_df.empty:
        raise SystemExit("No valid WebShop next-step monitoring settings found.")

    summary_df = (
        main_df.groupby(["adapter", "feature_group"], as_index=False)
        .agg(
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_auprc_gain=("auprc_gain", "mean"),
            mean_recall_at_5pct=("recall_at_5pct", "mean"),
            mean_precision_at_5pct=("precision_at_5pct", "mean"),
            total_positives=("positives", "sum"),
            total_n=("n", "sum"),
            valid_settings=("feature_group", "size"),
        )
    )
    by_model_df = (
        main_df.groupby(["model", "adapter", "ttc", "feature_group"], as_index=False)
        .agg(
            mean_auroc=("auroc", "mean"),
            mean_auprc=("auprc", "mean"),
            mean_auprc_gain=("auprc_gain", "mean"),
            mean_recall_at_5pct=("recall_at_5pct", "mean"),
            mean_precision_at_5pct=("precision_at_5pct", "mean"),
            total_positives=("positives", "sum"),
            total_n=("n", "sum"),
            valid_settings=("feature_group", "size"),
        )
    )

    main_df.to_csv(args.outdir / "next_step_prediction_main.csv", index=False)
    summary_df.to_csv(args.outdir / "next_step_prediction_summary.csv", index=False)
    by_model_df.to_csv(args.outdir / "next_step_prediction_by_model.csv", index=False)

    plot_feature_group(summary_df, args.outdir, "mean_auprc_gain", "fig_webshop_auprc_gain_by_feature_group.png", "Mean AUPRC gain")
    plot_feature_group(summary_df, args.outdir, "mean_recall_at_5pct", "fig_webshop_recall5_by_feature_group.png", "Mean Recall@5%")
    plot_by_model(by_model_df, args.outdir)
    if not args.aggregate_ttc:
        plot_bad_buy_rate_vs_ttc(df, args.outdir)
        plot_metric_vs_ttc(main_df, args.outdir, "internal_all", "auprc_gain", "fig_auprc_gain_vs_ttc.png", "AUPRC gain vs TTC")
        plot_signal_vs_ttc(
            df,
            args.outdir,
            ["reasoning_p_hack_late_change", "reasoning_p_hack_late_slope", "reasoning_p_hack"],
            "fig_late_stage_phack_vs_ttc.png",
            "Late-stage p(hack) vs TTC",
        )
        plot_metric_vs_ttc(main_df, args.outdir, "internal_all", "recall_at_5pct", "fig_recall5_vs_ttc.png", "Recall@5% vs TTC")

    print(f"Loaded rows: {len(df)}")
    print(f"Valid settings: {len(main_df)}")
    print("This analysis supports predictive monitoring if reasoning-phase internal features improve next-step prediction beyond the base rate under held-out episodes. It does not establish causality.")


if __name__ == "__main__":
    main()
