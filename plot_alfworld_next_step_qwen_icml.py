#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAIN_ADAPTERS = ["control", "mix10", "mix50", "hack"]
ALL_ADAPTERS = ["control", "mix05", "mix10", "mix50", "hack"]
ADAPTER_LABELS = {
    "control": "Control",
    "mix05": "Mix05",
    "mix10": "Mix10",
    "mix50": "Mix50",
    "hack": "Hack",
}
COLORS = {
    "bad": "#d62728",
    "nonbad": "#4c78a8",
    "entropy_group": "#1f77b4",
    "phack_group": "#d62728",
    "temporal_group": "#2ca02c",
    "combined_group": "#9467bd",
}
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


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.edgecolor": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_steps(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["model"].astype(str).str.startswith("Qwen")].copy()
    df = df[df["method"] == "react_ttc_monitored"].copy()
    df = df[df["adapter"].isin(ALL_ADAPTERS)].copy()
    df["ttc_num"] = pd.to_numeric(df["ttc"], errors="coerce")
    df["step_num"] = pd.to_numeric(df["step"], errors="coerce")
    df["bad_action"] = pd.to_numeric(df["bad_action"], errors="coerce")
    df = df.sort_values(["source_file", "step_num"]).reset_index(drop=True)
    group = df.groupby("source_file", sort=False)
    df["next_bad_action"] = group["bad_action"].shift(-1)
    df["has_next_step"] = group["step_num"].shift(-1).notna()
    df = df[df["has_next_step"]].copy()
    return df


def get_bin_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix + "_bin")]
    cols.sort(key=lambda c: int(c.split("bin")[-1]))
    return cols


def bootstrap_episode_curves(
    df: pd.DataFrame,
    cols: list[str],
    target_col: str,
    n_boot: int = 300,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    episodes = sorted(df["source_file"].astype(str).unique())
    bad_curves = []
    nonbad_curves = []
    bad_count = 0
    nonbad_count = 0

    for ep in episodes:
        ep_rows = df[df["source_file"].astype(str) == ep]
        bad_mask = ep_rows[target_col].fillna(0).astype(float) > 0.5
        nonbad_mask = ~bad_mask
        if bad_mask.any():
            arr = ep_rows.loc[bad_mask, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            bad_curves.append(np.nanmean(arr, axis=0))
            bad_count += int(bad_mask.sum())
        if nonbad_mask.any():
            arr = ep_rows.loc[nonbad_mask, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            nonbad_curves.append(np.nanmean(arr, axis=0))
            nonbad_count += int(nonbad_mask.sum())

    if not bad_curves:
        bad_curves = [np.full(len(cols), np.nan)]
    if not nonbad_curves:
        nonbad_curves = [np.full(len(cols), np.nan)]

    bad_curves = np.array(bad_curves, dtype=float)
    nonbad_curves = np.array(nonbad_curves, dtype=float)
    bad_mean = np.nanmean(bad_curves, axis=0)
    nonbad_mean = np.nanmean(nonbad_curves, axis=0)
    delta_mean = bad_mean - nonbad_mean

    rng = np.random.default_rng(seed)
    boot_bad = []
    boot_nonbad = []
    boot_delta = []
    for _ in range(n_boot):
        bad_idx = rng.integers(0, len(bad_curves), size=len(bad_curves))
        nonbad_idx = rng.integers(0, len(nonbad_curves), size=len(nonbad_curves))
        b = np.nanmean(bad_curves[bad_idx], axis=0)
        n = np.nanmean(nonbad_curves[nonbad_idx], axis=0)
        boot_bad.append(b)
        boot_nonbad.append(n)
        boot_delta.append(b - n)
    boot_bad = np.array(boot_bad)
    boot_nonbad = np.array(boot_nonbad)
    boot_delta = np.array(boot_delta)
    bad_lo, bad_hi = np.nanpercentile(boot_bad, [2.5, 97.5], axis=0)
    nonbad_lo, nonbad_hi = np.nanpercentile(boot_nonbad, [2.5, 97.5], axis=0)
    delta_lo, delta_hi = np.nanpercentile(boot_delta, [2.5, 97.5], axis=0)
    return (
        bad_mean,
        bad_lo,
        bad_hi,
        nonbad_mean,
        nonbad_lo,
        nonbad_hi,
        bad_count,
        nonbad_count,
    ), (delta_mean, delta_lo, delta_hi)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(X: np.ndarray, y: np.ndarray, steps: int = 500, lr: float = 0.12, l2: float = 1e-4) -> np.ndarray:
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


def prefix_features(row: pd.Series, n_bins: int, bin_idx: int) -> list[float]:
    e = np.array([pd.to_numeric(row[f"reasoning_entropy_bin{i}"], errors="coerce") for i in range(bin_idx + 1)], dtype=float)
    p = np.array([pd.to_numeric(row[f"reasoning_p_bin{i}"], errors="coerce") for i in range(bin_idx + 1)], dtype=float)
    e = np.nan_to_num(e, nan=np.nanmean(e) if np.isfinite(np.nanmean(e)) else 0.0)
    p = np.nan_to_num(p, nan=np.nanmean(p) if np.isfinite(np.nanmean(p)) else 0.0)

    def slope(vals: np.ndarray) -> float:
        if len(vals) < 2:
            return 0.0
        xs = np.linspace(0.0, 1.0, len(vals))
        xbar = float(xs.mean())
        ybar = float(vals.mean())
        denom = float(np.sum((xs - xbar) ** 2))
        if denom <= 0:
            return 0.0
        return float(np.sum((xs - xbar) * (vals - ybar)) / denom)

    return [
        float(e[-1]),
        float(p[-1]),
        float(np.mean(e)),
        float(np.mean(p)),
        slope(e),
        slope(p),
        float(e[-1] - e[0]),
        float(p[-1] - p[0]),
    ]


def compute_prefix_risk(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows = []
    for ttc in sorted(df["ttc_num"].dropna().unique()):
        sub = df[df["ttc_num"] == float(ttc)].copy()
        if sub.empty:
            continue
        y = sub["next_bad_action"].fillna(0).astype(float).to_numpy()
        groups = sub.groupby("source_file", sort=False)
        for bin_idx in range(n_bins):
            X = np.array([prefix_features(row, n_bins, bin_idx) for _, row in sub.iterrows()], dtype=float)
            oof = np.full(len(sub), np.nan, dtype=float)
            for _, idx in groups.indices.items():
                mask_test = np.zeros(len(sub), dtype=bool)
                mask_test[list(idx)] = True
                X_train = X[~mask_test]
                y_train = y[~mask_test]
                X_test = X[mask_test]
                if len(np.unique(y_train)) < 2:
                    continue
                mu = X_train.mean(axis=0)
                sigma = X_train.std(axis=0)
                sigma[sigma == 0.0] = 1.0
                X_train_z = (X_train - mu) / sigma
                X_test_z = (X_test - mu) / sigma
                w = fit_logreg(X_train_z, y_train)
                oof[mask_test] = predict_logreg(X_test_z, w)
            sub_bin = sub.copy()
            sub_bin["progress_bin"] = bin_idx
            sub_bin["normalized_progress"] = 0.0 if n_bins == 1 else bin_idx / (n_bins - 1)
            sub_bin["predicted_bad_risk"] = oof
            rows.append(sub_bin[["source_file", "adapter", "ttc_num", "progress_bin", "normalized_progress", "predicted_bad_risk"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_risk(risk_df: pd.DataFrame, adapter: str, ttc: int, n_boot: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = risk_df[(risk_df["adapter"] == adapter) & (risk_df["ttc_num"] == float(ttc))].copy()
    if sub.empty:
        nan = np.full(len(sorted(risk_df["progress_bin"].dropna().unique())), np.nan)
        return nan, nan, nan
    pivot = sub.pivot_table(index="source_file", columns="progress_bin", values="predicted_bad_risk", aggfunc="mean")
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    arr = pivot.to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        boots.append(np.nanmean(arr[idx], axis=0))
    boots = np.array(boots)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return mean, lo, hi


def save(fig: plt.Figure, png: Path) -> None:
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def row_limits(curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, float]:
    lo = math.inf
    hi = -math.inf
    for _, (mean, low, high) in curves.items():
        valid_lo = np.concatenate([mean[np.isfinite(mean)], low[np.isfinite(low)]]) if np.any(np.isfinite(low)) else mean[np.isfinite(mean)]
        valid_hi = np.concatenate([mean[np.isfinite(mean)], high[np.isfinite(high)]]) if np.any(np.isfinite(high)) else mean[np.isfinite(mean)]
        if len(valid_lo):
            lo = min(lo, float(np.nanmin(valid_lo)))
        if len(valid_hi):
            hi = max(hi, float(np.nanmax(valid_hi)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (-0.1, 0.1)
    pad = 0.05 * max(hi - lo, 1e-3)
    return lo - pad, hi + pad


def draw_delta_figure(df: pd.DataFrame, outdir: Path, adapters: list[str], stem: str, title: str) -> None:
    ent_cols = get_bin_columns(df, "reasoning_entropy")
    p_cols = get_bin_columns(df, "reasoning_p")
    n_bins = len(ent_cols)
    fig, axes = plt.subplots(2, len(adapters), figsize=(3.0 * len(adapters), 5.8), sharex=True, constrained_layout=True)
    if len(adapters) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    x = np.linspace(0.0, 1.0, n_bins)
    ent_cache = {}
    p_cache = {}
    counts = {}
    for adapter in adapters:
        sub = df[df["adapter"] == adapter]
        raw_ent, delta_ent = bootstrap_episode_curves(sub, ent_cols, "next_bad_action")
        raw_p, delta_p = bootstrap_episode_curves(sub, p_cols, "next_bad_action")
        ent_cache[adapter] = delta_ent
        p_cache[adapter] = delta_p
        counts[adapter] = (raw_ent[6], raw_ent[7])
    ent_lim = row_limits(ent_cache)
    p_lim = row_limits(p_cache)
    for c, adapter in enumerate(adapters):
        for r in range(2):
            axes[r, c].axvspan(0.8, 1.0, color="#ececec", zorder=0)
            axes[r, c].axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
            axes[r, c].grid(axis="y", alpha=0.18, linewidth=0.6)
            axes[r, c].set_xlim(0.0, 1.0)
        nb, ng = counts[adapter]
        axes[0, c].set_title(f"{ADAPTER_LABELS[adapter]}\n$n_{{bad}}$={nb}, $n_{{nonbad}}$={ng}")
        mean, lo, hi = ent_cache[adapter]
        axes[0, c].plot(x, mean, color=COLORS["bad"], linewidth=2.2)
        axes[0, c].fill_between(x, lo, hi, color=COLORS["bad"], alpha=0.18)
        mean, lo, hi = p_cache[adapter]
        axes[1, c].plot(x, mean, color=COLORS["bad"], linewidth=2.2)
        axes[1, c].fill_between(x, lo, hi, color=COLORS["bad"], alpha=0.18)
        axes[1, c].set_xlabel("Normalized reasoning progress within step $t$")
    axes[0, 0].set_ylabel(r"$\Delta$ reasoning entropy")
    axes[1, 0].set_ylabel(r"$\Delta$ reasoning $p_{hack}$")
    for c in range(len(adapters)):
        axes[0, c].set_ylim(*ent_lim)
        axes[1, c].set_ylim(*p_lim)
    fig.suptitle(title, y=1.05, fontsize=13)
    save(fig, outdir / f"{stem}.png")


def draw_raw_figure(df: pd.DataFrame, outdir: Path, adapters: list[str], stem: str, title: str) -> None:
    ent_cols = get_bin_columns(df, "reasoning_entropy")
    p_cols = get_bin_columns(df, "reasoning_p")
    n_bins = len(ent_cols)
    fig, axes = plt.subplots(2, len(adapters), figsize=(3.0 * len(adapters), 5.8), sharex=True, constrained_layout=True)
    if len(adapters) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    x = np.linspace(0.0, 1.0, n_bins)
    ent_cache_bad = {}
    ent_cache_nonbad = {}
    p_cache_bad = {}
    p_cache_nonbad = {}
    counts = {}
    for adapter in adapters:
        sub = df[df["adapter"] == adapter]
        raw_ent, _ = bootstrap_episode_curves(sub, ent_cols, "next_bad_action")
        raw_p, _ = bootstrap_episode_curves(sub, p_cols, "next_bad_action")
        ent_cache_bad[adapter] = raw_ent[:3]
        ent_cache_nonbad[adapter] = raw_ent[3:6]
        p_cache_bad[adapter] = raw_p[:3]
        p_cache_nonbad[adapter] = raw_p[3:6]
        counts[adapter] = (raw_ent[6], raw_ent[7])
    ent_lim = row_limits({k: v for k, v in ent_cache_bad.items()} | {k + "_n": v for k, v in ent_cache_nonbad.items()})
    p_lim = row_limits({k: v for k, v in p_cache_bad.items()} | {k + "_n": v for k, v in p_cache_nonbad.items()})
    for c, adapter in enumerate(adapters):
        for r in range(2):
            axes[r, c].axvspan(0.8, 1.0, color="#ececec", zorder=0)
            axes[r, c].grid(axis="y", alpha=0.18, linewidth=0.6)
            axes[r, c].set_xlim(0.0, 1.0)
        nb, ng = counts[adapter]
        axes[0, c].set_title(f"{ADAPTER_LABELS[adapter]}\n$n_{{bad}}$={nb}, $n_{{nonbad}}$={ng}")
        for ax, bad_triplet, good_triplet in [
            (axes[0, c], ent_cache_bad[adapter], ent_cache_nonbad[adapter]),
            (axes[1, c], p_cache_bad[adapter], p_cache_nonbad[adapter]),
        ]:
            mean, lo, hi = bad_triplet
            ax.plot(x, mean, color=COLORS["bad"], linewidth=2.0, label=r"$y^{bad}_{t+1}=1$")
            ax.fill_between(x, lo, hi, color=COLORS["bad"], alpha=0.18)
            mean, lo, hi = good_triplet
            ax.plot(x, mean, color=COLORS["nonbad"], linewidth=2.0, label=r"$y^{bad}_{t+1}=0$")
            ax.fill_between(x, lo, hi, color=COLORS["nonbad"], alpha=0.16)
        axes[1, c].set_xlabel("Normalized reasoning progress within step $t$")
    axes[0, 0].set_ylabel("Reasoning entropy")
    axes[1, 0].set_ylabel(r"Reasoning $p_{hack}$")
    for c in range(len(adapters)):
        axes[0, c].set_ylim(*ent_lim)
        axes[1, c].set_ylim(*p_lim)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, y=1.05, fontsize=13)
    save(fig, outdir / f"{stem}.png")


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
    prev = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(r - prev, 0.0)
        prev = r
    return float(ap)


def evaluate_prediction_vs_ttc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ttc in sorted(df["ttc_num"].dropna().unique()):
        sub = df[df["ttc_num"] == float(ttc)].copy()
        for adapter in ALL_ADAPTERS:
            da = sub[sub["adapter"] == adapter].copy()
            if da.empty:
                continue
            y = da["next_bad_action"].fillna(0).astype(float).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            for group_name, feats in FEATURE_GROUPS.items():
                X = da[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                oof = np.full(len(da), np.nan, dtype=float)
                groups = da.groupby("source_file", sort=False)
                for _, idx in groups.indices.items():
                    mask_test = np.zeros(len(da), dtype=bool)
                    mask_test[list(idx)] = True
                    X_train = X[~mask_test]
                    y_train = y[~mask_test]
                    X_test = X[mask_test]
                    if len(np.unique(y_train)) < 2:
                        continue
                    mu = np.nanmean(X_train, axis=0)
                    X_train = np.where(np.isnan(X_train), mu, X_train)
                    X_test = np.where(np.isnan(X_test), mu, X_test)
                    sigma = np.nanstd(X_train, axis=0)
                    sigma[sigma == 0.0] = 1.0
                    w = fit_logreg((X_train - mu) / sigma, y_train)
                    oof[mask_test] = predict_logreg((X_test - mu) / sigma, w)
                valid = ~np.isnan(oof)
                yv = y[valid]
                sv = oof[valid]
                if len(yv) == 0:
                    continue
                base = float(np.mean(yv))
                ap = average_precision(yv, sv)
                rows.append(
                    {
                        "ttc": int(ttc),
                        "adapter": adapter,
                        "feature_group": group_name,
                        "auroc": auc_score(yv, sv),
                        "ap_gain": ap - base if np.isfinite(ap) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def draw_prediction_vs_ttc(pred: pd.DataFrame, outdir: Path) -> None:
    if pred.empty:
        print("[WARN] No prediction-vs-TTC rows.")
        return
    summary = pred.groupby(["ttc", "feature_group"], as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_sem=("auroc", lambda s: float(pd.Series(s).sem()) if len(s) > 1 else 0.0),
        ap_gain_mean=("ap_gain", "mean"),
        ap_gain_sem=("ap_gain", lambda s: float(pd.Series(s).sem()) if len(s) > 1 else 0.0),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    for ax, ymean, ysem, ylabel, ref in [
        (axes[0], "auroc_mean", "auroc_sem", "AUROC", 0.5),
        (axes[1], "ap_gain_mean", "ap_gain_sem", "AUPRC gain over base rate", 0.0),
    ]:
        for group_name, feats in FEATURE_GROUPS.items():
            sub = summary[summary["feature_group"] == group_name].sort_values("ttc")
            ax.errorbar(
                sub["ttc"],
                sub[ymean],
                yerr=sub[ysem],
                marker="o",
                linewidth=2.0,
                color=COLORS[
                    "entropy_group"
                    if group_name == "entropy-only"
                    else "phack_group"
                    if group_name == "p_hack-only"
                    else "temporal_group"
                    if group_name == "temporal-p_hack-only"
                    else "combined_group"
                ],
                label=group_name,
            )
        ax.axhline(ref, color="#777777", linestyle="--", linewidth=1.0)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_xlabel("TTC")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Qwen next-step prediction vs test-time compute", y=1.05, fontsize=13)
    save(fig, outdir / "fig_alfworld_prediction_vs_ttc_qwen.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-csv", type=Path, default=Path("BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("BALROG/figures"))
    args = ap.parse_args()

    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_steps(args.steps_csv)
    print(f"[INFO] Loaded {len(df)} Qwen monitored step rows with next-step labels")
    df32 = df[df["ttc_num"] == 32.0].copy()
    n_bins = len(get_bin_columns(df, "reasoning_entropy"))
    risk_df = compute_prefix_risk(df, n_bins)
    print(f"[INFO] Computed prefix-risk trajectories: {len(risk_df)} rows")

    draw_delta_figure(
        df32[df32["adapter"].isin(MAIN_ADAPTERS)],
        args.outdir,
        MAIN_ADAPTERS,
        "fig_alfworld_next_step_temporal_separability_qwen_ttc32",
        "Internal trajectories at step $t$ separate future bad and non-bad actions",
    )
    draw_raw_figure(
        df32[df32["adapter"].isin(MAIN_ADAPTERS)],
        args.outdir,
        MAIN_ADAPTERS,
        "fig_alfworld_next_step_temporal_raw_qwen_ttc32",
        "Raw internal trajectories at step $t$ conditioned on bad action at $t{+}1$",
    )
    draw_delta_figure(
        df32[df32["adapter"].isin(ALL_ADAPTERS)],
        args.outdir,
        ALL_ADAPTERS,
        "fig_alfworld_next_step_temporal_separability_qwen_ttc32_appendix",
        "Appendix: Qwen temporal separability including Mix05",
    )
    pred = evaluate_prediction_vs_ttc(df)
    draw_prediction_vs_ttc(pred, args.outdir)
    print("[DONE] Generated next-step temporal figures in", args.outdir)


if __name__ == "__main__":
    main()
