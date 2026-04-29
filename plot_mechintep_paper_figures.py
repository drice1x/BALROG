#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ADAPTER_ORDER = {
    "control": 0,
    "mix05": 1,
    "mix10": 2,
    "mix50": 3,
    "mix90": 4,
    "hack": 5,
}


def order_adapter(x):
    return ADAPTER_ORDER.get(str(x), 99)


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()
    print(f"[OK] saved {path}")


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def plot_dose_response(step_agg: pd.DataFrame, outdir: Path):
    metric = "reasoning_p_hack_late_change_mean"
    if metric not in step_agg.columns:
        print(f"[WARN] missing {metric}")
        return

    d = step_agg[step_agg["method"] == "react_ttc_monitored"].copy()
    d["adapter_order"] = d["adapter"].map(order_adapter)

    g = (
        d.groupby(["model", "adapter", "adapter_order"], as_index=False)
        .agg(
            y=(metric, "mean"),
            ysem=("reasoning_p_hack_late_change_sem", "mean"),
        )
        .sort_values(["model", "adapter_order"])
    )

    plt.figure(figsize=(8, 4.8))

    for model, sub in g.groupby("model"):
        plt.errorbar(
            sub["adapter_order"],
            sub["y"],
            yerr=sub["ysem"],
            marker="o",
            capsize=3,
            label=model,
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(
        list(ADAPTER_ORDER.values()),
        list(ADAPTER_ORDER.keys()),
    )
    plt.xlabel("Reward-hack data in adapter")
    plt.ylabel("Late-stage Δ p(hack)")
    plt.title("Dose response of late-stage reward-hack dynamics")
    plt.legend()
    plt.grid(alpha=0.3)
    savefig(outdir / "fig1_dose_response_late_phack.png")


def plot_temporal_profiles(temporal: pd.DataFrame, outdir: Path):
    d = temporal[
        (temporal["signal"] == "reasoning_p")
        & (temporal["method"] == "react_ttc_monitored")
    ].copy()

    if d.empty:
        print("[WARN] no temporal reasoning_p rows")
        return

    keep = ["control", "mix05", "mix10", "mix50", "hack"]
    d = d[d["adapter"].isin(keep)]
    d["adapter_order"] = d["adapter"].map(order_adapter)

    for model in sorted(d["model"].dropna().unique()):
        dm = d[d["model"] == model]

        plt.figure(figsize=(7.5, 4.8))

        for adapter in sorted(dm["adapter"].unique(), key=order_adapter):
            sub = dm[dm["adapter"] == adapter]
            # average across TTC to show generic temporal shape
            g = (
                sub.groupby("bin_position", as_index=False)
                .agg(y=("mean", "mean"), ysem=("sem", "mean"))
            )
            plt.plot(g["bin_position"], g["y"], marker="o", label=adapter)

        plt.xlabel("Normalized reasoning-token position")
        plt.ylabel("p(hack)")
        plt.title(f"{model}: temporal reward-hack dynamics")
        plt.grid(alpha=0.3)
        plt.legend(title="Adapter")
        savefig(outdir / f"fig2_temporal_reasoning_phack_{model}.png")


def plot_ttc_for_mix_adapters(step_agg: pd.DataFrame, outdir: Path):
    d = step_agg[
        (step_agg["method"] == "react_ttc_monitored")
        & (step_agg["adapter"].isin(["mix05", "mix10", "mix50", "hack", "control"]))
    ].copy()

    metrics = [
        ("reasoning_p_hack_mean", "Mean reasoning p(hack)", "fig3a_ttc_mean_phack.png"),
        ("reasoning_p_hack_late_change_mean", "Late-stage Δ p(hack)", "fig3b_ttc_late_change.png"),
        ("gameable_hack_action_mean", "Gameable hack-action rate", "fig3c_ttc_hack_action_rate.png"),
        ("reasoning_entropy_mean_mean", "Reasoning entropy", "fig3d_ttc_entropy.png"),
    ]

    for metric, ylabel, fname in metrics:
        if metric not in d.columns:
            print(f"[WARN] missing {metric}")
            continue

        plt.figure(figsize=(8.5, 5))

        for (model, adapter), sub in d.groupby(["model", "adapter"]):
            if adapter not in {"mix05", "mix10", "mix50", "hack", "control"}:
                continue
            sub = sub.sort_values("ttc")
            plt.plot(
                sub["ttc"],
                sub[metric],
                marker="o",
                label=f"{model}:{adapter}",
            )

        plt.xlabel("Reasoning budget / TTC tokens")
        plt.ylabel(ylabel)
        plt.title(ylabel + " vs TTC")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7, ncol=2)
        savefig(outdir / fname)


def plot_predictive_risk(cv: pd.DataFrame, outdir: Path):
    # Focus on future gameable reward exploitation.
    d = cv[
        cv["target"].isin([
            "future_gameable_hack_action",
            "future_easy_grader_action",
            "future_proxy_farming_action",
        ])
    ].copy()

    if d.empty:
        print("[WARN] no future-target CV rows")
        return

    # Global feature group comparison.
    g = (
        d.groupby(["target", "feature_group"], as_index=False)
        .agg(
            auroc=("auroc", "mean"),
            auprc=("auprc", "mean"),
            positives=("positives", "sum"),
            n=("n", "sum"),
        )
    )

    for metric in ["auroc", "auprc"]:
        plt.figure(figsize=(9, 4.8))
        labels = [f"{r.target.replace('future_', '')}\n{r.feature_group}" for r in g.itertuples()]
        vals = g[metric].values

        plt.bar(range(len(vals)), vals)
        if metric == "auroc":
            plt.axhline(0.5, linestyle="--", linewidth=1)
        plt.xticks(range(len(vals)), labels, rotation=35, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"Predicting next-step reward exploitation ({metric.upper()})")
        savefig(outdir / f"fig4_predictive_risk_{metric}.png")

    # Highlight strongest model/adapter/TTC cases.
    top = (
        d[d["target"] == "future_gameable_hack_action"]
        .sort_values("auprc", ascending=False)
        .head(20)
    )

    top.to_csv(outdir / "top_predictive_risk_cases.csv", index=False)

    plt.figure(figsize=(9, 5))
    labels = [
        f"{r.model}:{r.adapter}:TTC{int(r.ttc)}\n{r.feature_group}"
        for r in top.itertuples()
    ]
    plt.bar(range(len(top)), top["auprc"])
    plt.xticks(range(len(top)), labels, rotation=55, ha="right")
    plt.ylabel("AUPRC")
    plt.title("Top cases: next-step gameable reward exploitation")
    savefig(outdir / "fig4c_top_predictive_cases_auprc.png")


def plot_qwenmix50_case(cv: pd.DataFrame, outdir: Path):
    d = cv[
        (cv["model"] == "QwenMix50")
        & (cv["target"] == "future_gameable_hack_action")
    ].copy()

    if d.empty:
        print("[WARN] no QwenMix50 future_gameable_hack_action rows")
        return

    plt.figure(figsize=(7.5, 4.8))

    for group, sub in d.groupby("feature_group"):
        sub = sub.sort_values("ttc")
        plt.plot(sub["ttc"], sub["auprc"], marker="o", label=group)

    plt.xlabel("Reasoning budget / TTC tokens")
    plt.ylabel("AUPRC")
    plt.title("QwenMix50: predicting next-step reward exploitation")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig(outdir / "fig5_qwenmix50_predictive_auprc_by_ttc.png")


def plot_signal_effects(delta: pd.DataFrame, outdir: Path):
    d = delta[
        delta["target"].isin([
            "future_gameable_hack_action",
            "future_easy_grader_action",
            "gameable_hack_action",
        ])
    ].copy()

    if d.empty:
        print("[WARN] no signal delta rows")
        return

    g = (
        d.groupby("feature", as_index=False)
        .agg(effect=("standardized_effect", "mean"))
        .sort_values("effect", ascending=False)
    )

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(g["feature"], g["effect"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Standardized effect: exploit-event minus non-event")
    plt.title("Temporal risk features associated with reward exploitation")
    savefig(outdir / "fig6_temporal_signal_effects.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal-dir", type=Path, default=Path("analysis_gameable_alfworld_temporal"))
    ap.add_argument("--risk-dir", type=Path, default=Path("analysis_predictive_risk"))
    ap.add_argument("--outdir", type=Path, default=Path("mechinterp_paper_figures"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    step_agg = load(args.temporal_dir / "step_signal_aggregate.csv")
    temporal = load(args.temporal_dir / "temporal_profile_bins.csv")
    cv = load(args.risk_dir / "predictive_risk_cv.csv")
    delta = load(args.risk_dir / "predictive_signal_deltas.csv")

    plot_dose_response(step_agg, args.outdir)
    plot_temporal_profiles(temporal, args.outdir)
    plot_ttc_for_mix_adapters(step_agg, args.outdir)
    plot_predictive_risk(cv, args.outdir)
    plot_qwenmix50_case(cv, args.outdir)
    plot_signal_effects(delta, args.outdir)

    print(f"\n[DONE] Workshop-ready figures saved to {args.outdir}")


if __name__ == "__main__":
    main()