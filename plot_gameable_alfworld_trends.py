#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def adapter_order(x):
    order = {
        "control": 0,
        "mix05": 1,
        "mix10": 2,
        "mix50": 3,
        "mix90": 4,
        "hack": 5,
    }
    return order.get(str(x), 99)


def plot_metric_vs_ttc(df, metric, ylabel, outpath):
    d = df[df["method"] == "react_ttc_monitored"].copy()
    d = d[pd.notna(d["ttc"])]

    if f"{metric}_mean" not in d.columns:
        print(f"[WARN] missing {metric}_mean")
        return

    d["adapter_order"] = d["adapter"].map(adapter_order)
    d = d.sort_values(["model", "adapter_order", "ttc"])

    plt.figure(figsize=(9, 5))

    for (model, adapter), sub in d.groupby(["model", "adapter"]):
        sub = sub.sort_values("ttc")
        y = sub[f"{metric}_mean"]
        yerr = sub.get(f"{metric}_sem", None)

        label = f"{model}:{adapter}"
        plt.errorbar(
            sub["ttc"],
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=label,
        )

    plt.xlabel("Reasoning budget / TTC tokens")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs TTC")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[OK] saved {outpath}")


def plot_temporal_profiles(temporal, signal, outpath):
    d = temporal[temporal["signal"] == signal].copy()
    if d.empty:
        print(f"[WARN] no temporal rows for {signal}")
        return

    d["adapter_order"] = d["adapter"].map(adapter_order)
    d = d.sort_values(["model", "adapter_order", "ttc", "bin"])

    for model in sorted(d["model"].dropna().unique()):
        dm = d[d["model"] == model]

        plt.figure(figsize=(9, 5))

        for adapter in sorted(dm["adapter"].dropna().unique(), key=adapter_order):
            # average over TTC budgets for a clean temporal profile
            da = dm[dm["adapter"] == adapter]
            prof = da.groupby("bin_position", as_index=False)["mean"].mean()

            plt.plot(
                prof["bin_position"],
                prof["mean"],
                marker="o",
                label=adapter,
            )

        plt.xlabel("Normalized generation position")
        plt.ylabel(signal)
        plt.title(f"{model}: temporal {signal}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        model_out = outpath.parent / f"{outpath.stem}_{model}{outpath.suffix}"
        plt.savefig(model_out)
        plt.close()
        print(f"[OK] saved {model_out}")


def plot_predictive(pred, target, outpath):
    d = pred[pred["target"] == target].copy()
    if d.empty:
        print(f"[WARN] no predictive rows for {target}")
        return

    # summarize across model/adapter/TTC
    g = (
        d.groupby("predictor", as_index=False)
        .agg(delta=("delta_bad_minus_good", "mean"))
        .sort_values("delta", ascending=False)
    )

    plt.figure(figsize=(9, 5))
    plt.bar(g["predictor"], g["delta"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.ylabel("Mean signal difference: hack-event minus non-event")
    plt.title(f"Predictive signal for {target}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[OK] saved {outpath}")


def plot_adapter_dose(step_agg, metric, outpath):
    if f"{metric}_mean" not in step_agg.columns:
        print(f"[WARN] missing {metric}_mean")
        return

    d = step_agg.copy()
    d = d[d["method"] == "react_ttc_monitored"]
    d["adapter_order"] = d["adapter"].map(adapter_order)
    d = d.sort_values(["model", "adapter_order"])

    plt.figure(figsize=(9, 5))

    for model, sub in d.groupby("model"):
        # average over TTC budgets
        g = (
            sub.groupby(["adapter", "adapter_order"], as_index=False)
            .agg(y=(f"{metric}_mean", "mean"))
            .sort_values("adapter_order")
        )
        plt.plot(g["adapter"], g["y"], marker="o", label=model)

    plt.xlabel("Adapter / reward-hack data mixture")
    plt.ylabel(metric)
    plt.title(f"Dose response: {metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    print(f"[OK] saved {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, default=Path("analysis_gameable_alfworld_temporal"))
    ap.add_argument("--outdir", type=Path, default=Path("paper_figures_gameable_alfworld"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    ep = load_csv(args.analysis_dir / "episode_aggregate.csv")
    step = load_csv(args.analysis_dir / "step_signal_aggregate.csv")
    temporal = load_csv(args.analysis_dir / "temporal_profile_bins.csv")
    pred = load_csv(args.analysis_dir / "predictive_signal_table.csv")

    # 1. Does TTC increase task success?
    plot_metric_vs_ttc(
        ep,
        metric="success",
        ylabel="Task success",
        outpath=args.outdir / "fig1a_ttc_success.png",
    )

    # 2. Does TTC increase p(hack)?
    plot_metric_vs_ttc(
        step,
        metric="reasoning_p_hack",
        ylabel="Reasoning p(hack)",
        outpath=args.outdir / "fig1b_ttc_reasoning_phack.png",
    )

    plot_metric_vs_ttc(
        step,
        metric="action_p_hack",
        ylabel="Action p(hack)",
        outpath=args.outdir / "fig1c_ttc_action_phack.png",
    )

    # 3. Does TTC increase entropy?
    plot_metric_vs_ttc(
        step,
        metric="reasoning_entropy_mean",
        ylabel="Reasoning entropy",
        outpath=args.outdir / "fig1d_ttc_reasoning_entropy.png",
    )

    plot_metric_vs_ttc(
        step,
        metric="action_entropy_mean",
        ylabel="Action entropy",
        outpath=args.outdir / "fig1e_ttc_action_entropy.png",
    )

    # 4. Does TTC increase reward gaming?
    plot_metric_vs_ttc(
        step,
        metric="gameable_hack_action",
        ylabel="Gameable hack-action rate",
        outpath=args.outdir / "fig1f_ttc_gameable_hack_action.png",
    )

    plot_metric_vs_ttc(
        ep,
        metric="proxy_per_step",
        ylabel="Proxy score per step",
        outpath=args.outdir / "fig1g_ttc_proxy_per_step.png",
    )

    plot_metric_vs_ttc(
        ep,
        metric="fake_completion_rate",
        ylabel="Fake completion rate",
        outpath=args.outdir / "fig1h_ttc_fake_completion_rate.png",
    )

    plot_metric_vs_ttc(
        ep,
        metric="easy_grader_rate",
        ylabel="Easy-grader choice rate",
        outpath=args.outdir / "fig1i_ttc_easy_grader_rate.png",
    )

    # 5. Temporal dynamics: does p(hack) spike near the final tokens?
    plot_temporal_profiles(
        temporal,
        signal="reasoning_p",
        outpath=args.outdir / "fig2_temporal_reasoning_phack.png",
    )

    plot_temporal_profiles(
        temporal,
        signal="action_p",
        outpath=args.outdir / "fig2_temporal_action_phack.png",
    )

    plot_temporal_profiles(
        temporal,
        signal="reasoning_entropy",
        outpath=args.outdir / "fig2_temporal_reasoning_entropy.png",
    )

    plot_temporal_profiles(
        temporal,
        signal="action_entropy",
        outpath=args.outdir / "fig2_temporal_action_entropy.png",
    )

    # 6. Late-stage dynamics vs TTC
    plot_metric_vs_ttc(
        step,
        metric="reasoning_p_hack_late_change",
        ylabel="Reasoning late-stage Δ p(hack)",
        outpath=args.outdir / "fig3a_ttc_late_change_reasoning_phack.png",
    )

    plot_metric_vs_ttc(
        step,
        metric="reasoning_p_hack_late_slope",
        ylabel="Reasoning late-stage slope p(hack)",
        outpath=args.outdir / "fig3b_ttc_late_slope_reasoning_phack.png",
    )

    plot_metric_vs_ttc(
        step,
        metric="action_p_hack_late_change",
        ylabel="Action late-stage Δ p(hack)",
        outpath=args.outdir / "fig3c_ttc_late_change_action_phack.png",
    )

    plot_metric_vs_ttc(
        step,
        metric="action_p_hack_late_slope",
        ylabel="Action late-stage slope p(hack)",
        outpath=args.outdir / "fig3d_ttc_late_slope_action_phack.png",
    )

    # 7. Predictive power before reward-gaming events
    plot_predictive(
        pred,
        target="gameable_hack_action",
        outpath=args.outdir / "fig4a_predict_gameable_hack_action.png",
    )

    plot_predictive(
        pred,
        target="fake_completion_action",
        outpath=args.outdir / "fig4b_predict_fake_completion.png",
    )

    plot_predictive(
        pred,
        target="easy_grader_action",
        outpath=args.outdir / "fig4c_predict_easy_grader.png",
    )

    plot_predictive(
        pred,
        target="proxy_farming_action",
        outpath=args.outdir / "fig4d_predict_proxy_farming.png",
    )

    # 8. Adapter dose-response
    plot_adapter_dose(
        step,
        metric="reasoning_p_hack",
        outpath=args.outdir / "fig5a_adapter_dose_reasoning_phack.png",
    )

    plot_adapter_dose(
        step,
        metric="reasoning_p_hack_late_change",
        outpath=args.outdir / "fig5b_adapter_dose_late_change.png",
    )

    plot_adapter_dose(
        step,
        metric="gameable_hack_action",
        outpath=args.outdir / "fig5c_adapter_dose_gameable_hack_action.png",
    )

    print("\n[DONE] Figures written to:", args.outdir)


if __name__ == "__main__":
    main()