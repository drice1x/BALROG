#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TTC_FULL = [8, 16, 32, 64, 128, 256]
TTC_COMPACT = [16, 32, 64, 128]
ADAPTER_ORDER = ["control", "mix05", "mix10", "mix50", "hack"]
ADAPTER_LABELS = {
    "control": "Control",
    "mix05": "Mix05",
    "mix10": "Mix10",
    "mix50": "Mix50",
    "hack": "Hack",
}
ADAPTER_COLORS = {
    "control": "#4c78a8",
    "mix05": "#54a24b",
    "mix10": "#f58518",
    "mix50": "#e45756",
    "hack": "#b279a2",
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


def load_steps(path: Path, family: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["model"].astype(str).str.startswith(family)].copy()
    df = df[df["method"] == "react_ttc_monitored"].copy()
    df = df[df["adapter"].isin(ADAPTER_ORDER)].copy()
    df["ttc_num"] = pd.to_numeric(df["ttc"], errors="coerce")
    df["step_num"] = pd.to_numeric(df["step"], errors="coerce")
    df["bad_action"] = pd.to_numeric(df["bad_action"], errors="coerce")
    df = df.sort_values(["source_file", "step_num"]).reset_index(drop=True)
    group = df.groupby("source_file", sort=False)
    df["next_bad_action"] = group["bad_action"].shift(-1)
    df["has_next_step"] = group["step_num"].shift(-1).notna()
    df = df[df["has_next_step"]].copy()
    return df


def sem_across_rows(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.full(8, np.nan)
    n = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    sem[n <= 1] = np.nan
    return sem


def summarize_metric(df: pd.DataFrame, prefix: str, adapter: str, ttc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df[(df["adapter"] == adapter) & (df["ttc_num"] == float(ttc))]
    cols = [f"{prefix}_bin{i}" for i in range(8)]
    arr = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if arr.size == 0:
        x = np.linspace(0, 1, 8)
        nan = np.full(8, np.nan)
        return x, nan, nan
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(arr, axis=0)
    sem = sem_across_rows(arr)
    return np.linspace(0, 1, 8), mean, 1.96 * sem


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


def prefix_features(row: pd.Series, bin_idx: int) -> list[float]:
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


def compute_prefix_risk(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ttc in TTC_FULL:
        sub = df[df["ttc_num"] == float(ttc)].copy()
        if sub.empty:
            continue
        y = sub["next_bad_action"].fillna(0).astype(float).to_numpy()
        groups = sub.groupby("source_file", sort=False)
        for bin_idx in range(8):
            X = np.array([prefix_features(row, bin_idx) for _, row in sub.iterrows()], dtype=float)
            oof = np.full(len(sub), np.nan, dtype=float)
            for source_file, idx in groups.indices.items():
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
            sub_bin["normalized_progress"] = bin_idx / 7.0
            sub_bin["predicted_bad_risk"] = oof
            rows.append(sub_bin[["source_file", "adapter", "ttc_num", "progress_bin", "normalized_progress", "predicted_bad_risk"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_risk(risk_df: pd.DataFrame, adapter: str, ttc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = risk_df[(risk_df["adapter"] == adapter) & (risk_df["ttc_num"] == float(ttc))].copy()
    if sub.empty:
        nan = np.full(8, np.nan)
        return np.linspace(0, 1, 8), nan, nan
    pivot = sub.pivot_table(index="source_file", columns="progress_bin", values="predicted_bad_risk", aggfunc="mean")
    pivot = pivot.reindex(columns=list(range(8)))
    arr = pivot.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(arr, axis=0)
    sem = sem_across_rows(arr)
    return np.linspace(0, 1, 8), mean, 1.96 * sem


def row_limits_from_series(series_map: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[float, float]:
    lo = math.inf
    hi = -math.inf
    for _, (_, mean, ci) in series_map.items():
        valid_lo = mean - ci
        valid_hi = mean + ci
        if np.any(np.isfinite(valid_lo)):
            lo = min(lo, float(np.nanmin(valid_lo)))
        if np.any(np.isfinite(valid_hi)):
            hi = max(hi, float(np.nanmax(valid_hi)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0)
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    return (lo - pad, hi + pad)


def draw_figure(df: pd.DataFrame, risk_df: pd.DataFrame, ttcs: list[int], out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(3, len(ttcs), figsize=(2.7 * len(ttcs), 7.2), sharex=True, constrained_layout=True)
    if len(ttcs) == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    entropy_cache = {}
    phack_cache = {}
    risk_cache = {}
    for ttc in ttcs:
        for adapter in ADAPTER_ORDER:
            entropy_cache[(ttc, adapter)] = summarize_metric(df, "reasoning_entropy", adapter, ttc)
            phack_cache[(ttc, adapter)] = summarize_metric(df, "reasoning_p", adapter, ttc)
            risk_cache[(ttc, adapter)] = summarize_risk(risk_df, adapter, ttc)

    ent_lim = row_limits_from_series(entropy_cache)
    ph_lim = row_limits_from_series(phack_cache)
    risk_lim = row_limits_from_series(risk_cache)

    row_labels = [
        "Reasoning entropy",
        "Reasoning $p_{hack}$",
        "Predicted risk of bad action at $t{+}1$",
    ]

    for c, ttc in enumerate(ttcs):
        for r in range(3):
            axes[r, c].axvspan(0.8, 1.0, color="#ececec", zorder=0)
            axes[r, c].grid(axis="y", alpha=0.18, linewidth=0.6)
            axes[r, c].set_xlim(0.0, 1.0)
        axes[0, c].set_title(f"TTC = {ttc}")

        for adapter in ADAPTER_ORDER:
            color = ADAPTER_COLORS[adapter]
            for ax, cache in [
                (axes[0, c], entropy_cache),
                (axes[1, c], phack_cache),
                (axes[2, c], risk_cache),
            ]:
                x, mean, ci = cache[(ttc, adapter)]
                ax.plot(x, mean, color=color, linewidth=2.0, label=ADAPTER_LABELS[adapter])
                ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.16)

    for c in range(len(ttcs)):
        axes[0, c].set_ylim(*ent_lim)
        axes[1, c].set_ylim(*ph_lim)
        axes[2, c].set_ylim(max(0.0, risk_lim[0]), min(1.0, risk_lim[1]))
        axes[2, c].set_xlabel("Normalized reasoning progress within step $t$")

    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(title, y=1.06, fontsize=14)
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-csv", type=Path, default=Path("BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("BALROG/figures"))
    ap.add_argument("--family", type=str, default="Qwen", choices=["Qwen", "Falcon", "Llama"])
    args = ap.parse_args()

    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading step trajectories from {args.steps_csv}")
    df = load_steps(args.steps_csv, args.family)
    print(f"[INFO] Loaded {len(df)} {args.family} monitored step rows with next-step labels")
    risk_df = compute_prefix_risk(df)
    print(f"[INFO] Computed prefix-risk trajectories: {len(risk_df)} rows")

    stem = f"{args.family.lower()}_internal_reasoning_dynamics_ttc"
    full_png = args.outdir / f"{stem}_full.png"
    compact_png = args.outdir / f"{stem}_compact.png"

    draw_figure(
        df,
        risk_df,
        TTC_FULL,
        full_png,
        f"{args.family} internal reasoning dynamics across test-time compute",
    )
    draw_figure(
        df,
        risk_df,
        TTC_COMPACT,
        compact_png,
        f"{args.family} internal reasoning dynamics across test-time compute",
    )

    print("[DONE] Generated figures:")
    print(" ", full_png)
    print(" ", full_png.with_suffix(".pdf"))
    print(" ", compact_png)
    print(" ", compact_png.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
