#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ADAPTER_ORDER = ["control", "mix05", "mix10", "mix50", "hack"]
QWEN_MODEL_ORDER = ["QwenControl", "QwenMix05", "QwenMix10", "QwenMix50", "QwenHack"]
FAMILY_ORDER = ["Qwen", "Llama", "Falcon"]
FEATURE_GROUPS = {
    "entropy-only": ["reasoning_entropy_mean", "action_entropy_mean"],
    "p_hack-only": ["reasoning_p_hack", "action_p_hack"],
    "temporal-p_hack-only": [
        "reasoning_p_hack_late_change",
        "reasoning_p_hack_late_slope",
        "action_p_hack_late_change",
        "action_p_hack_late_slope",
    ],
    "combined": [
        "reasoning_entropy_mean",
        "action_entropy_mean",
        "reasoning_p_hack",
        "action_p_hack",
        "reasoning_p_hack_late_change",
        "reasoning_p_hack_late_slope",
        "action_p_hack_late_change",
        "action_p_hack_late_slope",
    ],
}
FEATURE_COLORS = {
    "entropy-only": "#1f77b4",
    "p_hack-only": "#d62728",
    "temporal-p_hack-only": "#2ca02c",
    "combined": "#9467bd",
}
BAD_COLOR = "#d62728"
GOOD_COLOR = "#4c78a8"
GOOD_FILL = "#9aa0a6"
ADAPTER_COLORS = {
    "control": "#1f77b4",
    "mix05": "#2ca02c",
    "mix10": "#ff7f0e",
    "mix50": "#d62728",
    "hack": "#9467bd",
}
MARKER_MAP = {"always": "o", "gated": "s", "none": "o"}


@dataclass
class SourceFiles:
    steps: Path | None
    predictive: Path | None
    step_agg: Path | None
    steering_agg: Path | None
    steering_effects: Path | None


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.grid": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def family_of_model(model: str) -> str:
    text = str(model)
    for fam in FAMILY_ORDER:
        if text.startswith(fam):
            return fam
    return "Other"


def locate(root: Path, relative_candidates: list[str]) -> Path | None:
    for rel in relative_candidates:
        path = root / rel
        if path.exists():
            return path
    for rel in relative_candidates:
        target = Path(rel).name
        matches = list(root.rglob(target))
        if matches:
            return matches[0]
    return None


def discover_sources(root: Path) -> SourceFiles:
    return SourceFiles(
        steps=locate(root, ["analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"]),
        predictive=locate(root, ["analysis_gameable_alfworld_temporal/predictive_signal_table.csv"]),
        step_agg=locate(root, ["analysis_gameable_alfworld_temporal/step_signal_aggregate.csv"]),
        steering_agg=locate(
            root,
            [
                "hf_alfworld_steering/hf_steering_analysis_mix_sweep/aggregate_by_condition.csv",
                "hf_alfworld_steering/hf_steering_analysis_mix_sweep_qwen/aggregate_by_condition.csv",
            ],
        ),
        steering_effects=locate(
            root,
            [
                "hf_alfworld_steering/hf_steering_analysis_mix_sweep/steering_effects.csv",
                "hf_alfworld_steering/hf_steering_analysis_mix_sweep_qwen/steering_effects.csv",
            ],
        ),
    )


def load_df(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_float(x):
    try:
        if x is None or x == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def add_next_step_targets(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["step_num"] = pd.to_numeric(d["step"], errors="coerce")
    d = d.sort_values(["source_file", "step_num"]).reset_index(drop=True)
    group = d.groupby("source_file", sort=False)
    for t in targets:
        d[f"next_{t}"] = group[t].shift(-1)
    d["has_next_step"] = group["step_num"].shift(-1).notna()
    return d


def summarize_binned_curves(df: pd.DataFrame, prefix: str, mask: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = [f"{prefix}_bin{i}" for i in range(8) if f"{prefix}_bin{i}" in df.columns]
    arr = df.loc[mask, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        x = np.linspace(0.0, 1.0, len(cols))
        empty = np.full(len(cols), np.nan)
        return x, empty, empty
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(arr, axis=0)
        n = np.sum(~np.isnan(arr), axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    sem[n <= 1] = np.nan
    ci = 1.96 * sem
    x = np.linspace(0.0, 1.0, len(cols))
    return x, mean, ci


def savefig(fig: plt.Figure, png_path: Path) -> None:
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _plot_temporal_grid(
    steps: pd.DataFrame,
    model_order: list[str],
    out: Path,
    title: str,
    generated: list[str],
) -> None:
    q = steps[
        (steps["family"] == "Qwen")
        & (steps["method"] == "react_ttc_monitored")
        & (steps["ttc_num"] == 32.0)
        & (steps["has_next_step"])
    ].copy()
    if q.empty:
        print("[WARN] No Qwen TTC=32 step data for temporal figure.")
        return

    ncols = len(model_order)
    fig, axes = plt.subplots(2, ncols, figsize=(3.15 * ncols, 6.0), sharex=True, constrained_layout=True)
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    row_limits = {0: [np.inf, -np.inf], 1: [np.inf, -np.inf]}
    counts: dict[str, tuple[int, int]] = {}

    # First pass to get shared y-limits.
    cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for model in model_order:
        sub = q[q["model"] == model]
        bad_mask = sub["next_bad_action"].fillna(0).astype(float) > 0.5
        good_mask = sub["next_bad_action"].fillna(0).astype(float) <= 0.5
        counts[model] = (int(bad_mask.sum()), int(good_mask.sum()))
        rp = summarize_binned_curves(sub, "reasoning_p", bad_mask)
        rg = summarize_binned_curves(sub, "reasoning_p", good_mask)
        ep = summarize_binned_curves(sub, "reasoning_entropy", bad_mask)
        eg = summarize_binned_curves(sub, "reasoning_entropy", good_mask)
        cache[(model, "p")] = (*rp, *rg)
        cache[(model, "e")] = (*ep, *eg)
        for idx, payload in enumerate(((*rp, *rg), (*ep, *eg))):
            _, mean_bad, ci_bad, _, mean_good, ci_good = payload
            vals_lo = np.concatenate([(mean_bad - ci_bad)[np.isfinite(mean_bad - ci_bad)], (mean_good - ci_good)[np.isfinite(mean_good - ci_good)]])
            vals_hi = np.concatenate([(mean_bad + ci_bad)[np.isfinite(mean_bad + ci_bad)], (mean_good + ci_good)[np.isfinite(mean_good + ci_good)]])
            if len(vals_lo):
                row_limits[idx][0] = min(row_limits[idx][0], float(np.nanmin(vals_lo)))
            if len(vals_hi):
                row_limits[idx][1] = max(row_limits[idx][1], float(np.nanmax(vals_hi)))

    for col, model in enumerate(model_order):
        short = model.replace("Qwen", "")
        ax_p = axes[0, col]
        ax_e = axes[1, col]
        ax_p.axvspan(0.8, 1.0, color="#eaeaea", zorder=0)
        ax_e.axvspan(0.8, 1.0, color="#eaeaea", zorder=0)

        for ax, key in [
            (ax_p, "p"),
            (ax_e, "e"),
        ]:
            xb, mean_bad, ci_bad, xg, mean_good, ci_good = cache[(model, key)]
            ax.plot(xb, mean_bad, color=BAD_COLOR, linewidth=2.1, label=r"$y^{bad}_{t+1}=1$")
            ax.fill_between(xb, mean_bad - ci_bad, mean_bad + ci_bad, color=BAD_COLOR, alpha=0.18)
            ax.plot(xg, mean_good, color=GOOD_COLOR, linewidth=2.1, label=r"$y^{bad}_{t+1}=0$")
            ax.fill_between(xg, mean_good - ci_good, mean_good + ci_good, color=GOOD_FILL, alpha=0.18)
            ax.set_xlim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.18, linewidth=0.6)

        nb, ng = counts[model]
        ax_p.set_title(f"{short}\n$n_{{bad}}$={nb}, $n_{{nonbad}}$={ng}")
        ax_e.set_xlabel("Normalized reasoning progress within step $t$")
        if col == 0:
            ax_p.set_ylabel("Reasoning $p_{hack}$")
            ax_e.set_ylabel("Reasoning entropy")

    for col in range(ncols):
        if np.isfinite(row_limits[0][0]) and np.isfinite(row_limits[0][1]):
            axes[0, col].set_ylim(row_limits[0][0] - 0.02, row_limits[0][1] + 0.02)
        if np.isfinite(row_limits[1][0]) and np.isfinite(row_limits[1][1]):
            axes[1, col].set_ylim(row_limits[1][0] - 0.02, row_limits[1][1] + 0.02)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, y=1.07, fontsize=13)
    savefig(fig, out)
    generated.extend([str(out), str(out.with_suffix(".pdf"))])


def plot_main_temporal_trajectories(steps: pd.DataFrame, outdir: Path, generated: list[str]) -> None:
    out = outdir / "alfworld_temporal_trajectories_qwen_ttc32.png"
    _plot_temporal_grid(
        steps,
        QWEN_MODEL_ORDER,
        out,
        "Internal reasoning trajectories at step $t$ predict next-step action risk",
        generated,
    )


def plot_compact_temporal_trajectories(steps: pd.DataFrame, outdir: Path, generated: list[str]) -> None:
    out = outdir / "alfworld_temporal_trajectories_qwen_ttc32_compact.png"
    _plot_temporal_grid(
        steps,
        ["QwenControl", "QwenMix10", "QwenHack"],
        out,
        "Internal reasoning trajectories at step $t$ predict next-step action risk",
        generated,
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(X: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.1, l2: float = 1e-4) -> np.ndarray:
    w = np.zeros(X.shape[1] + 1, dtype=float)
    Xb = np.column_stack([np.ones(len(X)), X])
    for _ in range(steps):
        p = _sigmoid(Xb @ w)
        grad = (Xb.T @ (p - y)) / len(X)
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w


def predict_logreg(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    return _sigmoid(Xb @ w)


def auc_score(y: np.ndarray, scores: np.ndarray) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    if np.sum(y == 1) == 0:
        return float("nan")
    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.sum(y == 1)
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(r - prev_recall, 0.0)
        prev_recall = r
    return float(ap)


def evaluate_feature_groups(steps: pd.DataFrame, ttc: float | None = 32.0, family: str = "Qwen") -> pd.DataFrame:
    d = steps[(steps["family"] == family) & (steps["method"] == "react_ttc_monitored") & (steps["has_next_step"])].copy()
    if ttc is not None:
        d = d[d["ttc_num"] == float(ttc)]
    rows = []
    for model in sorted(d["model"].unique(), key=lambda m: ADAPTER_ORDER.index(d[d["model"] == m]["adapter"].iloc[0])):
        dm = d[d["model"] == model].copy()
        y = dm["next_bad_action"].fillna(0).astype(float).to_numpy()
        groups = dm.groupby("source_file", sort=False)
        for group_name, feats in FEATURE_GROUPS.items():
            oof_scores = np.full(len(dm), np.nan, dtype=float)
            for source_file, test_idx in groups.indices.items():
                mask_test = np.zeros(len(dm), dtype=bool)
                mask_test[list(test_idx)] = True
                train = dm.loc[~mask_test, feats].apply(pd.to_numeric, errors="coerce")
                test = dm.loc[mask_test, feats].apply(pd.to_numeric, errors="coerce")
                y_train = y[~mask_test]
                if len(np.unique(y_train)) < 2:
                    continue
                mu = train.mean(axis=0)
                train = train.fillna(mu)
                test = test.fillna(mu)
                sigma = train.std(axis=0).replace(0.0, 1.0).fillna(1.0)
                X_train = ((train - mu) / sigma).to_numpy(dtype=float)
                X_test = ((test - mu) / sigma).to_numpy(dtype=float)
                w = fit_logreg(X_train, y_train)
                oof_scores[mask_test] = predict_logreg(X_test, w)
            valid = ~np.isnan(oof_scores)
            yv = y[valid]
            sv = oof_scores[valid]
            base_rate = float(np.mean(yv)) if len(yv) else float("nan")
            ap = average_precision(yv, sv) if len(yv) else float("nan")
            rows.append(
                {
                    "model": model,
                    "adapter": dm["adapter"].iloc[0],
                    "feature_group": group_name,
                    "n": int(np.sum(valid)),
                    "positives": int(np.sum(yv == 1)),
                    "base_rate": base_rate,
                    "auroc": auc_score(yv, sv) if len(yv) else float("nan"),
                    "ap": ap,
                    "ap_gain": ap - base_rate if np.isfinite(ap) and np.isfinite(base_rate) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def plot_prediction_summary(steps: pd.DataFrame, outdir: Path, generated: list[str]) -> pd.DataFrame:
    metrics = evaluate_feature_groups(steps, ttc=32.0, family="Qwen")
    if metrics.empty:
        print("[WARN] No Qwen metrics for Figure 2.")
        return metrics
    metrics["adapter_order"] = metrics["adapter"].map(lambda a: ADAPTER_ORDER.index(a))
    metrics = metrics.sort_values("adapter_order")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True, sharey=True)
    y_positions = np.arange(len(QWEN_MODEL_ORDER))
    offsets = np.linspace(-0.24, 0.24, len(FEATURE_GROUPS))

    for ax, metric_col, xlabel in [
        (axes[0], "ap_gain", "AUPRC gain over base rate"),
        (axes[1], "auroc", "AUROC for predicting bad_action$_{t+1}$"),
    ]:
        for offset, (group_name, _) in zip(offsets, FEATURE_GROUPS.items()):
            sub = metrics[metrics["feature_group"] == group_name]
            xs = []
            ys = []
            for i, model in enumerate(QWEN_MODEL_ORDER):
                row = sub[sub["model"] == model]
                if row.empty:
                    continue
                xs.append(float(row.iloc[0][metric_col]))
                ys.append(y_positions[i] + offset)
            ax.scatter(xs, ys, s=36, color=FEATURE_COLORS[group_name], label=group_name, zorder=3)
        ax.grid(axis="x", alpha=0.2, linewidth=0.6)
        ax.set_xlabel(xlabel)
    axes[0].axvline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    axes[1].axvline(0.5, color="#888888", linestyle="--", linewidth=1.0)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([m.replace("Qwen", "Qwen ") for m in QWEN_MODEL_ORDER])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Next-step predictiveness is adapter-dependent", y=1.06, fontsize=13)
    axes[0].text(
        0.02,
        -0.16,
        "Below 0 means worse than base rate.",
        transform=axes[0].transAxes,
        fontsize=8,
        color="#444444",
    )
    axes[1].text(
        0.02,
        -0.16,
        "Below 0.5 means not predictive.",
        transform=axes[1].transAxes,
        fontsize=8,
        color="#444444",
    )
    out = outdir / "alfworld_prediction_summary_qwen.png"
    savefig(fig, out)
    generated.extend([str(out), str(out.with_suffix(".pdf"))])
    return metrics


def plot_steering_frontier(steering_effects: pd.DataFrame, outdir: Path, generated: list[str]) -> None:
    if steering_effects.empty:
        print("[WARN] No steering effects for Figure 3.")
        return
    d = steering_effects[steering_effects["model_family"].str.lower() == "qwen"].copy()
    d = d[d["adapter"].isin(["mix05", "mix10", "mix50"])].copy()
    if d.empty:
        print("[WARN] No Qwen steering rows for Figure 3.")
        return

    for col in [
        "delta_steered_minus_unsteered_proxy_score_without_success",
        "steering_rate_steered_mean",
        "delta_steered_minus_unsteered_easy_grader_count",
    ]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["exploit_reduction"] = -d["delta_steered_minus_unsteered_proxy_score_without_success"]
    d["easy_reduction"] = -d["delta_steered_minus_unsteered_easy_grader_count"]

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), constrained_layout=True)
    for _, row in d.iterrows():
        ax.scatter(
            row["steering_rate_steered_mean"],
            row["exploit_reduction"],
            s=60,
            color=ADAPTER_COLORS[row["adapter"]],
            marker=MARKER_MAP.get(row["steering_mode"], "o"),
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
    best = d.sort_values(["exploit_reduction", "steering_rate_steered_mean"], ascending=[False, True]).head(3)
    for _, row in best.iterrows():
        ax.annotate(
            f"{row['adapter']} {row['steering_mode']}",
            (row["steering_rate_steered_mean"], row["exploit_reduction"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Fraction of steps steered")
    ax.set_ylabel("Reduction in proxy score without true success")
    ax.grid(alpha=0.2, linewidth=0.6)
    handles = []
    from matplotlib.lines import Line2D
    for adapter in ["mix05", "mix10", "mix50"]:
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=ADAPTER_COLORS[adapter], markeredgecolor="black", label=adapter, markersize=7))
    handles.append(Line2D([0], [0], marker="o", color="black", linestyle="None", label="always-on", markersize=7))
    handles.append(Line2D([0], [0], marker="s", color="black", linestyle="None", label="gated", markersize=7))
    ax.legend(handles=handles, frameon=False, loc="best", ncol=2)
    out = outdir / "alfworld_steering_frontier_qwen.png"
    savefig(fig, out)
    generated.extend([str(out), str(out.with_suffix(".pdf"))])


def plot_all_model_temporal_appendix(steps: pd.DataFrame, outdir: Path, generated: list[str]) -> None:
    d = steps[
        (steps["method"] == "react_ttc_monitored")
        & (steps["ttc_num"] == 32.0)
        & (steps["has_next_step"])
        & (steps["adapter"].isin(ADAPTER_ORDER))
    ].copy()
    if d.empty:
        print("[WARN] No step data for appendix trajectories.")
        return
    fig, axes = plt.subplots(6, 5, figsize=(16.5, 13.0), sharex=True, constrained_layout=True)
    row_specs = []
    for fam in FAMILY_ORDER:
        row_specs.append((fam, "reasoning_p", "Reasoning $p_{hack}$"))
        row_specs.append((fam, "reasoning_entropy", "Reasoning entropy"))
    x = np.linspace(0.0, 1.0, 8)
    for r, (family, prefix, ylabel) in enumerate(row_specs):
        for c, adapter in enumerate(ADAPTER_ORDER):
            ax = axes[r, c]
            sub = d[(d["family"] == family) & (d["adapter"] == adapter)]
            ax.axvspan(0.8, 1.0, color="#efefef", zorder=0)
            if not sub.empty:
                bad_mask = sub["next_bad_action"].fillna(0).astype(float) > 0.5
                good_mask = ~bad_mask
                xb, mean_bad, ci_bad = summarize_binned_curves(sub, prefix, bad_mask)
                xg, mean_good, ci_good = summarize_binned_curves(sub, prefix, good_mask)
                ax.plot(xb, mean_bad, color="#d62728", linewidth=1.5)
                ax.fill_between(xb, mean_bad - ci_bad, mean_bad + ci_bad, color="#d62728", alpha=0.16)
                ax.plot(xg, mean_good, color="#1f77b4", linewidth=1.5)
                ax.fill_between(xg, mean_good - ci_good, mean_good + ci_good, color="#1f77b4", alpha=0.16)
            if r == 0:
                ax.set_title(adapter)
            if c == 0:
                ax.set_ylabel(f"{family}\n{ylabel}")
            if r == len(row_specs) - 1:
                ax.set_xlabel("Norm. progress")
            ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Appendix: temporal trajectories across all model families at TTC=32", y=1.02, fontsize=13)
    out = outdir / "alfworld_temporal_trajectories_all_models_ttc32.png"
    savefig(fig, out)
    generated.extend([str(out), str(out.with_suffix(".pdf"))])


def plot_qwen_ttc_sweep(steps: pd.DataFrame, outdir: Path, generated: list[str]) -> None:
    d = steps[(steps["family"] == "Qwen") & (steps["method"] == "react_ttc_monitored") & (steps["has_next_step"])].copy()
    if d.empty:
        print("[WARN] No Qwen step data for TTC sweep.")
        return
    rows = []
    for ttc in [8, 16, 32, 64, 128, 256]:
        metrics = evaluate_feature_groups(steps, ttc=float(ttc), family="Qwen")
        if metrics.empty:
            continue
        metrics["ttc"] = ttc
        rows.append(metrics)
    if not rows:
        return
    m = pd.concat(rows, ignore_index=True)
    pooled = (
        m.groupby(["ttc", "feature_group"], as_index=False)
        .agg(auroc=("auroc", "mean"), ap_gain=("ap_gain", "mean"))
        .sort_values("ttc")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    for ax, ycol, ylabel, baseline in [
        (axes[0], "auroc", "Mean AUROC across Qwen adapters", 0.5),
        (axes[1], "ap_gain", "Mean AUPRC gain across Qwen adapters", 0.0),
    ]:
        for group_name in FEATURE_GROUPS:
            sub = pooled[pooled["feature_group"] == group_name]
            ax.plot(sub["ttc"], sub[ycol], marker="o", linewidth=2.0, color=FEATURE_COLORS[group_name], label=group_name)
        ax.axhline(baseline, color="#888888", linestyle="--", linewidth=1.0)
        ax.set_xlabel("TTC")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Appendix: reasoning budget changes next-step predictiveness", y=1.05, fontsize=13)
    out = outdir / "alfworld_ttc_sweep_qwen.png"
    savefig(fig, out)
    generated.extend([str(out), str(out.with_suffix(".pdf"))])


def plot_delta_heatmap_appendix(predictive: pd.DataFrame, outdir: Path, generated: list[str]) -> None:
    d = predictive[
        (predictive["method"] == "react_ttc_monitored")
        & (predictive["ttc_num"] == 32.0)
        & (predictive["target"] == "bad_action")
        & (predictive["predictor"].isin([
            "reasoning_entropy_mean",
            "action_entropy_mean",
            "reasoning_p_hack",
            "action_p_hack",
            "reasoning_p_hack_late_change",
            "action_p_hack_late_change",
        ]))
    ].copy()
    if d.empty:
        print("[WARN] No predictive rows for appendix heatmap.")
        return
    model_order = sorted(d["model"].unique(), key=lambda m: (FAMILY_ORDER.index(family_of_model(m)), ADAPTER_ORDER.index(d[d["model"] == m]["adapter"].iloc[0])))
    pred_order = [
        "reasoning_entropy_mean",
        "action_entropy_mean",
        "reasoning_p_hack",
        "action_p_hack",
        "reasoning_p_hack_late_change",
        "action_p_hack_late_change",
    ]
    mat = d.pivot_table(index="model", columns="predictor", values="delta_bad_minus_good", aggfunc="mean").reindex(index=model_order, columns=pred_order)
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    vmax = np.nanmax(np.abs(mat.to_numpy(dtype=float)))
    vmax = 0.1 if not np.isfinite(vmax) or vmax == 0 else vmax
    im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pred_order)))
    ax.set_xticklabels(["R-ent", "A-ent", "R-p", "A-p", "R-lateΔ", "A-lateΔ"], rotation=35, ha="right")
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7)
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="#555555")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Δ(bad_{t+1} minus non-bad_{t+1})")
    ax.set_title("Appendix: signed next-step delta heatmap at TTC=32")
    out = outdir / "alfworld_delta_heatmap_all_models_ttc32.png"
    savefig(fig, out)
    generated.extend([str(out), str(out.with_suffix(".pdf"))])


def write_readme(outdir: Path, sources: SourceFiles, generated: list[str], qwen_metrics: pd.DataFrame) -> None:
    lines = [
        "# ALFWorld ICML Figure Suite",
        "",
        "## Source files found",
        "",
        f"- steps_with_entropy_phack_temporal.csv: `{sources.steps}`",
        f"- predictive_signal_table.csv: `{sources.predictive}`",
        f"- step_signal_aggregate.csv: `{sources.step_agg}`",
        f"- steering aggregate: `{sources.steering_agg}`",
        f"- steering effects: `{sources.steering_effects}`",
        "",
        "## Figures generated",
        "",
    ]
    for g in generated:
        lines.append(f"- `{g}`")
    lines.extend(
        [
            "",
        "## Figure logic",
        "",
        "1. `alfworld_temporal_trajectories_qwen_ttc32` is the main hero figure.",
        "   It shows within-step reasoning trajectories at step `t`, conditioned on whether `bad_action_{t+1}` occurs.",
        "2. `alfworld_temporal_trajectories_qwen_ttc32_compact` is the compact main-paper variant.",
        "3. `alfworld_prediction_summary_qwen` summarizes next-step predictive strength of step-`t` feature groups.",
        "4. `alfworld_steering_frontier_qwen` connects these signals to intervention usefulness in Gameable ALFWorld.",
        "5. Appendix figures extend the same story across families, TTC budgets, and signed deltas.",
        "",
        "## Qwen prediction summary snapshot",
        "",
        ]
    )
    if not qwen_metrics.empty:
        pivot = qwen_metrics.pivot_table(index="model", columns="feature_group", values="auroc", aggfunc="mean")
        cols = list(pivot.columns)
        lines.append("| Model | " + " | ".join(cols) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(cols)) + "|")
        for idx, row in pivot.iterrows():
            vals = []
            for col in cols:
                value = row[col]
                vals.append("" if pd.isna(value) else f"{float(value):.3f}")
            lines.append("| " + str(idx) + " | " + " | ".join(vals) + " |")
    (outdir / "README.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("BALROG"))
    ap.add_argument("--outdir", type=Path, default=Path("BALROG/figures"))
    args = ap.parse_args()

    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(args.repo_root)
    print("[INFO] Source discovery")
    for name, path in sources.__dict__.items():
        print(f"  {name}: {path}")

    steps = load_df(sources.steps)
    predictive = load_df(sources.predictive)
    steering_effects = load_df(sources.steering_effects)

    generated: list[str] = []
    if not steps.empty:
        steps["family"] = steps["model"].map(family_of_model)
        steps["ttc_num"] = pd.to_numeric(steps["ttc"], errors="coerce")
        target_cols = ["bad_action", "gameable_hack_action", "easy_grader_action", "fake_completion_action"]
        for col in target_cols:
            steps[col] = pd.to_numeric(steps[col], errors="coerce")
        steps = add_next_step_targets(steps, target_cols)

    if not predictive.empty:
        predictive["family"] = predictive["model"].map(family_of_model)
        predictive["ttc_num"] = pd.to_numeric(predictive["ttc"], errors="coerce")
        predictive["delta_bad_minus_good"] = pd.to_numeric(predictive["delta_bad_minus_good"], errors="coerce")

    plot_main_temporal_trajectories(steps, args.outdir, generated)
    qwen_metrics = plot_prediction_summary(steps, args.outdir, generated)
    plot_steering_frontier(steering_effects, args.outdir, generated)
    plot_all_model_temporal_appendix(steps, args.outdir, generated)
    plot_qwen_ttc_sweep(steps, args.outdir, generated)
    plot_delta_heatmap_appendix(predictive, args.outdir, generated)
    write_readme(args.outdir, sources, generated, qwen_metrics)

    print("[DONE] Generated figures:")
    for g in generated:
        print(" ", g)
    print(f"[DONE] README: {args.outdir / 'README.md'}")


if __name__ == "__main__":
    main()
