#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


TTCS = [8, 16, 32, 64, 128, 256]
ADAPTERS = ["control", "mix05", "mix10", "mix50", "hack"]
ADAPTER_LABELS = {
    "control": "Control",
    "mix05": "Mix05",
    "mix10": "Mix10",
    "mix50": "Mix50",
    "hack": "Hack",
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
FEATURE_COLORS = {
    "entropy-only": "#1f77b4",
    "p_hack-only": "#d62728",
    "temporal-p_hack-only": "#2ca02c",
    "combined": "#9467bd",
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


def load_steps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["model"].astype(str).str.startswith("Qwen")].copy()
    df = df[df["method"] == "react_ttc_monitored"].copy()
    df = df[df["adapter"].isin(ADAPTERS)].copy()
    df["ttc_num"] = pd.to_numeric(df["ttc"], errors="coerce")
    df["step_num"] = pd.to_numeric(df["step"], errors="coerce")
    df["bad_action"] = pd.to_numeric(df["bad_action"], errors="coerce")
    df = df.sort_values(["source_file", "step_num"]).reset_index(drop=True)
    group = df.groupby("source_file", sort=False)
    df["next_bad_action"] = group["bad_action"].shift(-1)
    df["has_next_step"] = group["step_num"].shift(-1).notna()
    return df[df["has_next_step"]].copy()


def get_bin_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix + "_bin")]
    cols.sort(key=lambda c: int(c.split("bin")[-1]))
    return cols


def bootstrap_delta(
    sub: pd.DataFrame,
    cols: list[str],
    target_col: str = "next_bad_action",
    n_boot: int = 300,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    episodes = sorted(sub["source_file"].astype(str).unique())
    bad_ep = []
    nonbad_ep = []
    n_bad = 0
    n_nonbad = 0
    for ep in episodes:
        ep_rows = sub[sub["source_file"].astype(str) == ep]
        bad_mask = ep_rows[target_col].fillna(0).astype(float) > 0.5
        nonbad_mask = ~bad_mask
        if bad_mask.any():
            arr = ep_rows.loc[bad_mask, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            bad_ep.append(np.nanmean(arr, axis=0))
            n_bad += int(bad_mask.sum())
        if nonbad_mask.any():
            arr = ep_rows.loc[nonbad_mask, cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            nonbad_ep.append(np.nanmean(arr, axis=0))
            n_nonbad += int(nonbad_mask.sum())
    if not bad_ep or not nonbad_ep:
        nan = np.full(len(cols), np.nan)
        return nan, nan, nan, n_bad, n_nonbad
    bad_ep = np.array(bad_ep, dtype=float)
    nonbad_ep = np.array(nonbad_ep, dtype=float)
    delta = np.nanmean(bad_ep, axis=0) - np.nanmean(nonbad_ep, axis=0)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        bi = rng.integers(0, len(bad_ep), size=len(bad_ep))
        ni = rng.integers(0, len(nonbad_ep), size=len(nonbad_ep))
        boots.append(np.nanmean(bad_ep[bi], axis=0) - np.nanmean(nonbad_ep[ni], axis=0))
    boots = np.array(boots)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return delta, lo, hi, n_bad, n_nonbad


def compute_heatmap_tables(df: pd.DataFrame) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], np.ndarray], dict[str, tuple[int, int]], int]:
    ent_cols = get_bin_cols(df, "reasoning_entropy")
    p_cols = get_bin_cols(df, "reasoning_p")
    n_bins = len(ent_cols)
    ent_map: dict[tuple[str, int], np.ndarray] = {}
    p_map: dict[tuple[str, int], np.ndarray] = {}
    counts: dict[str, tuple[int, int]] = {}
    for adapter in ADAPTERS:
        total_bad = 0
        total_nonbad = 0
        for ttc in TTCS:
            sub = df[(df["adapter"] == adapter) & (df["ttc_num"] == float(ttc))]
            ent_delta, _, _, n_bad, n_nonbad = bootstrap_delta(sub, ent_cols, seed=ttc + 1)
            p_delta, _, _, _, _ = bootstrap_delta(sub, p_cols, seed=ttc + 1001)
            ent_map[(adapter, ttc)] = ent_delta
            p_map[(adapter, ttc)] = p_delta
            total_bad += n_bad
            total_nonbad += n_nonbad
        counts[adapter] = (total_bad, total_nonbad)
    return ent_map, p_map, counts, n_bins


def save(fig: plt.Figure, path_png: Path) -> None:
    fig.savefig(path_png, dpi=240, bbox_inches="tight")
    fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_heatmaps(df: pd.DataFrame, outdir: Path) -> None:
    ent_map, p_map, counts, n_bins = compute_heatmap_tables(df)
    x_centers = np.linspace(0.0, 1.0, n_bins)
    extent = [0.0, 1.0, min(TTCS) - 4, max(TTCS) + 4]

    ent_stack = np.vstack([ent_map[(adapter, ttc)] for adapter in ADAPTERS for ttc in TTCS])
    p_stack = np.vstack([p_map[(adapter, ttc)] for adapter in ADAPTERS for ttc in TTCS])
    ent_v = float(np.nanmax(np.abs(ent_stack))) if np.any(np.isfinite(ent_stack)) else 0.1
    p_v = float(np.nanmax(np.abs(p_stack))) if np.any(np.isfinite(p_stack)) else 0.1

    fig, axes = plt.subplots(2, len(ADAPTERS), figsize=(16.2, 6.0), constrained_layout=True, sharex=True, sharey=True)
    for c, adapter in enumerate(ADAPTERS):
        ent_mat = np.vstack([ent_map[(adapter, ttc)] for ttc in TTCS])
        p_mat = np.vstack([p_map[(adapter, ttc)] for ttc in TTCS])
        nb, nn = counts[adapter]
        for r, (mat, vmax, label) in enumerate(
            [
                (ent_mat, ent_v, r"$\Delta$ reasoning entropy"),
                (p_mat, p_v, r"$\Delta$ reasoning $p_{hack}$"),
            ]
        ):
            ax = axes[r, c]
            norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="coolwarm", norm=norm, extent=extent)
            ax.axvline(0.8, color="#555555", linestyle="--", linewidth=1.0)
            ax.axvspan(0.8, 1.0, color="#cccccc", alpha=0.10, zorder=0)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticks(TTCS)
            if r == 0:
                ax.set_title(f"{ADAPTER_LABELS[adapter]}\n$n_{{bad}}$={nb}, $n_{{nonbad}}$={nn}")
            if c == 0:
                ax.set_ylabel(label + "\nTTC")
            if r == 1:
                ax.set_xlabel("Normalized reasoning progress within step $t$")
        # row-wise colorbars after loop
    cbar1 = fig.colorbar(axes[0, -1].images[0], ax=axes[0, :], shrink=0.92, pad=0.02)
    cbar1.set_label(r"$E[\cdot \mid bad_{t+1}=1] - E[\cdot \mid bad_{t+1}=0]$")
    cbar2 = fig.colorbar(axes[1, -1].images[0], ax=axes[1, :], shrink=0.92, pad=0.02)
    cbar2.set_label(r"$E[\cdot \mid bad_{t+1}=1] - E[\cdot \mid bad_{t+1}=0]$")
    fig.suptitle("Qwen next-step separability over reasoning progress and test-time compute", y=1.03, fontsize=13)
    save(fig, outdir / "fig_qwen_ttc_progress_heatmaps_nextstep.png")


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


def evaluate_prediction(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ttc in TTCS:
        sub = df[df["ttc_num"] == float(ttc)].copy()
        for adapter in ADAPTERS:
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
                if not np.any(valid):
                    continue
                yv = y[valid]
                sv = oof[valid]
                base = float(np.mean(yv))
                rows.append(
                    {
                        "ttc": ttc,
                        "adapter": adapter,
                        "feature_group": group_name,
                        "auroc": auc_score(yv, sv),
                        "ap_gain": average_precision(yv, sv) - base,
                    }
                )
    return pd.DataFrame(rows)


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


def draw_prediction_vs_ttc(df: pd.DataFrame, outdir: Path) -> None:
    pred = evaluate_prediction(df)
    if pred.empty:
        print("[WARN] No prediction rows.")
        return
    agg = pred.groupby(["ttc", "feature_group"], as_index=False).agg(
        auroc_mean=("auroc", "mean"),
        auroc_sem=("auroc", lambda s: float(pd.Series(s).sem()) if len(s) > 1 else 0.0),
        ap_gain_mean=("ap_gain", "mean"),
        ap_gain_sem=("ap_gain", lambda s: float(pd.Series(s).sem()) if len(s) > 1 else 0.0),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    for ax, ymean, ysem, ylabel, ref in [
        (axes[0], "auroc_mean", "auroc_sem", "AUROC", 0.5),
        (axes[1], "ap_gain_mean", "ap_gain_sem", "AUPRC gain over base rate", 0.0),
    ]:
        for group in FEATURE_GROUPS:
            sub = agg[agg["feature_group"] == group].sort_values("ttc")
            ax.errorbar(
                sub["ttc"],
                sub[ymean],
                yerr=sub[ysem],
                marker="o",
                linewidth=2.0,
                color=FEATURE_COLORS[group],
                label=group,
            )
        ax.axhline(ref, color="#777777", linestyle="--", linewidth=1.0)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_xlabel("TTC")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Qwen prediction vs TTC", y=1.05, fontsize=13)
    save(fig, outdir / "fig_qwen_prediction_vs_ttc_clean.png")


def draw_latewindow_vs_ttc(df: pd.DataFrame, outdir: Path) -> None:
    ent_cols = get_bin_cols(df, "reasoning_entropy")
    p_cols = get_bin_cols(df, "reasoning_p")
    k = max(1, int(math.ceil(0.2 * len(ent_cols))))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), constrained_layout=True)
    for ax, cols, ylabel in [
        (axes[0], ent_cols, r"Late-window $\Delta$ entropy"),
        (axes[1], p_cols, r"Late-window $\Delta$ $p_{hack}$"),
    ]:
        for adapter in ADAPTERS:
            means = []
            los = []
            his = []
            for ttc in TTCS:
                sub = df[(df["adapter"] == adapter) & (df["ttc_num"] == float(ttc))]
                delta, lo, hi, _, _ = bootstrap_delta(sub, cols, seed=ttc + len(adapter))
                means.append(np.nanmean(delta[-k:]))
                los.append(np.nanmean(lo[-k:]))
                his.append(np.nanmean(hi[-k:]))
            means = np.array(means, dtype=float)
            los = np.array(los, dtype=float)
            his = np.array(his, dtype=float)
            ax.plot(TTCS, means, marker="o", linewidth=2.0, color=ADAPTER_COLORS[adapter], label=ADAPTER_LABELS[adapter])
            ax.fill_between(TTCS, los, his, color=ADAPTER_COLORS[adapter], alpha=0.16)
        ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.set_xlabel("TTC")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Qwen late-window next-step separability vs TTC", y=1.05, fontsize=13)
    save(fig, outdir / "fig_qwen_latewindow_vs_ttc.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-csv", type=Path, default=Path("BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"))
    ap.add_argument("--outdir", type=Path, default=Path("BALROG/figures"))
    args = ap.parse_args()

    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_steps(args.steps_csv)
    print(f"[INFO] Loaded {len(df)} Qwen monitored step rows with next-step labels")
    draw_heatmaps(df, args.outdir)
    draw_prediction_vs_ttc(df, args.outdir)
    draw_latewindow_vs_ttc(df, args.outdir)
    print("[DONE] Generated Qwen TTC-progress heatmap figure set in", args.outdir)


if __name__ == "__main__":
    main()
