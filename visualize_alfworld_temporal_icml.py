#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FAMILY_ORDER = ["Qwen", "Llama", "Falcon"]
ADAPTER_ORDER = ["control", "mix05", "mix10", "mix50", "hack"]
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
PREDICTOR_LABELS = {
    "reasoning_entropy_mean": "Reasoning\nentropy",
    "action_entropy_mean": "Action\nentropy",
    "reasoning_p_hack": "Reasoning\np_hack",
    "action_p_hack": "Action\np_hack",
    "reasoning_p_hack_late_change": "Reasoning\nlate Δp_hack",
    "action_p_hack_late_change": "Action\nlate Δp_hack",
    "reasoning_p_hack_late_slope": "Reasoning\nlate slope",
    "action_p_hack_late_slope": "Action\nlate slope",
}
ADAPTER_COLORS = {
    "control": "#1f77b4",
    "mix05": "#2ca02c",
    "mix10": "#ff7f0e",
    "mix50": "#d62728",
    "hack": "#9467bd",
}


def adapter_key(adapter: str) -> int:
    try:
        return ADAPTER_ORDER.index(str(adapter))
    except ValueError:
        return 99


def family_of_model(model: str) -> str:
    text = str(model)
    for family in FAMILY_ORDER:
        if text.startswith(family):
            return family
    return "Other"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def filter_predictive_table(pred: pd.DataFrame, target: str, ttc: float) -> pd.DataFrame:
    d = pred.copy()
    d["ttc_num"] = pd.to_numeric(d["ttc"], errors="coerce")
    d["delta_bad_minus_good"] = pd.to_numeric(d["delta_bad_minus_good"], errors="coerce")
    d["n_bad"] = pd.to_numeric(d["n_bad"], errors="coerce")
    d["family"] = d["model"].map(family_of_model)
    d = d[
        (d["method"] == "react_ttc_monitored")
        & (d["target"] == target)
        & (d["ttc_num"] == float(ttc))
        & d["predictor"].isin(PREDICTOR_ORDER)
    ].copy()
    return d


def filter_step_table(step: pd.DataFrame, ttc: float) -> pd.DataFrame:
    d = step.copy()
    d["ttc_num"] = pd.to_numeric(d["ttc"], errors="coerce")
    d["family"] = d["model"].map(family_of_model)
    d = d[(d["method"] == "react_ttc_monitored") & (d["ttc_num"] == float(ttc))].copy()
    return d


def annotate_heatmap(ax, mat: pd.DataFrame) -> None:
    for i, row_name in enumerate(mat.index):
        for j, col_name in enumerate(mat.columns):
            value = mat.loc[row_name, col_name]
            if pd.isna(value):
                continue
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8, color="black")


def plot_family_heatmaps(pred: pd.DataFrame, outpath: Path, target: str, ttc: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    vmax = pred["delta_bad_minus_good"].abs().max()
    vmax = 0.1 if pd.isna(vmax) or vmax == 0 else float(vmax)

    for ax, family in zip(axes, FAMILY_ORDER):
        sub = pred[pred["family"] == family].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{family}\nno data")
            continue

        sub["adapter_order"] = sub["adapter"].map(adapter_key)
        sub = sub.sort_values(["adapter_order", "predictor"])
        mat = sub.pivot_table(
            index="adapter",
            columns="predictor",
            values="delta_bad_minus_good",
            aggfunc="mean",
        )
        mat = mat.reindex(index=ADAPTER_ORDER, columns=PREDICTOR_ORDER)
        im = ax.imshow(mat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([PREDICTOR_LABELS.get(c, c) for c in mat.columns], rotation=35, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_title(f"{family}: step-t signals before {target}_{{t+1}}\nTTC={int(ttc)}")
        annotate_heatmap(ax, mat)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Δ(bad-next-action minus non-bad-next-action)")
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_phack_tradeoff(step: pd.DataFrame, outpath: Path, ttc: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for ax, family in zip(axes, FAMILY_ORDER):
        sub = step[step["family"] == family].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{family}\nno data")
            continue

        sub["adapter_order"] = sub["adapter"].map(adapter_key)
        sub = sub.sort_values("adapter_order")

        x = pd.to_numeric(sub["reasoning_entropy_mean_mean"], errors="coerce")
        y = pd.to_numeric(sub["reasoning_p_hack_mean"], errors="coerce")
        s = 80 + 520 * pd.to_numeric(sub["bad_action_mean"], errors="coerce").fillna(0.0)

        for _, row in sub.iterrows():
            adapter = row["adapter"]
            ax.scatter(
                row["reasoning_entropy_mean_mean"],
                row["reasoning_p_hack_mean"],
                s=80 + 520 * float(row.get("bad_action_mean", 0.0) or 0.0),
                color=ADAPTER_COLORS.get(adapter, "#333333"),
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
            )
            ax.text(
                row["reasoning_entropy_mean_mean"] + 0.01,
                row["reasoning_p_hack_mean"] + 0.005,
                str(adapter),
                fontsize=8,
            )

        ax.plot(x, y, color="#777777", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("Mean reasoning entropy at step t")
        ax.set_ylabel("Mean reasoning p_hack at step t")
        ax.set_title(f"{family}: internal state vs harmful action rate\nTTC={int(ttc)}")
        ax.grid(alpha=0.25)

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_separability_scatter(pred: pd.DataFrame, outpath: Path, target: str, ttc: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for ax, family in zip(axes, FAMILY_ORDER):
        sub = pred[pred["family"] == family].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{family}\nno data")
            continue

        ent = (
            sub[sub["predictor"] == "reasoning_entropy_mean"][["adapter", "delta_bad_minus_good", "n_bad"]]
            .rename(columns={"delta_bad_minus_good": "entropy_delta"})
        )
        ph = (
            sub[sub["predictor"] == "reasoning_p_hack"][["adapter", "delta_bad_minus_good"]]
            .rename(columns={"delta_bad_minus_good": "phack_delta"})
        )
        merged = ent.merge(ph, on="adapter", how="inner")
        merged["adapter_order"] = merged["adapter"].map(adapter_key)
        merged = merged.sort_values("adapter_order")

        for _, row in merged.iterrows():
            adapter = row["adapter"]
            ax.scatter(
                row["entropy_delta"],
                row["phack_delta"],
                s=50 + 2.5 * float(row.get("n_bad", 0.0) or 0.0),
                color=ADAPTER_COLORS.get(adapter, "#333333"),
                edgecolors="black",
                linewidths=0.6,
                alpha=0.9,
            )
            ax.text(row["entropy_delta"] + 0.005, row["phack_delta"] + 0.005, str(adapter), fontsize=8)

        ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Δ reasoning entropy at t before bad_{t+1}")
        ax.set_ylabel("Δ reasoning p_hack at t before bad_{t+1}")
        ax.set_title(f"{family}: next-step separability\nTTC={int(ttc)}")
        ax.grid(alpha=0.25)

    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(pred: pd.DataFrame, step: pd.DataFrame, outpath: Path, target: str, ttc: float) -> None:
    target_rows = pred[pred["predictor"].isin(["reasoning_entropy_mean", "reasoning_p_hack"])].copy()
    lines = [
        "# ALFWorld Temporal Visualization Summary",
        "",
        f"- Target: `{target}`",
        f"- TTC: `{int(ttc)}`",
        "",
        "## Figure set",
        "",
        "- `fig_temporal_heatmaps.png`: family heatmaps of step-`t` signal deltas before `t+1` harmful actions",
        "- `fig_entropy_phack_tradeoff.png`: family scatterplots linking mean entropy, mean `p_hack`, and harmful-action rate",
        "- `fig_temporal_separability.png`: family scatterplots of entropy-vs-`p_hack` separability before `bad_action_{t+1}`",
        "",
        "## High-level interpretation",
        "",
    ]

    for family in FAMILY_ORDER:
        fam = target_rows[target_rows["family"] == family]
        if fam.empty:
            continue
        ent = fam[fam["predictor"] == "reasoning_entropy_mean"].sort_values("adapter")
        ph = fam[fam["predictor"] == "reasoning_p_hack"].sort_values("adapter")
        if ent.empty or ph.empty:
            continue
        best_idx = ph["delta_bad_minus_good"].abs().idxmax()
        best_ph = ph.loc[best_idx] if pd.notna(best_idx) else None
        lines.append(f"- `{family}`: next-step signal deltas are available for all five adapters.")
        if best_ph is not None:
            lines.append(
                f"  Strongest reasoning `p_hack` separation here is `{best_ph['adapter']}` "
                f"with `Δ={float(best_ph['delta_bad_minus_good']):+.3f}`."
            )

    lines.extend(
        [
            "",
            "## Paper use",
            "",
            "Use these figures in the ALFWorld results section to show that internal state at step `t` already",
            "contains measurable information about harmful action selection at step `t+1`, and that this relation",
            "can be compared across model families and adapter mixtures.",
            "",
        ]
    )
    outpath.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path, default=Path("BALROG/analysis_gameable_alfworld_temporal"))
    ap.add_argument("--outdir", type=Path, default=Path("BALROG/figures_alfworld_temporal_icml"))
    ap.add_argument("--target", type=str, default="bad_action")
    ap.add_argument("--ttc", type=float, default=32.0)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    pred = load_csv(args.analysis_dir / "predictive_signal_table.csv")
    step = load_csv(args.analysis_dir / "step_signal_aggregate.csv")
    pred = filter_predictive_table(pred, args.target, args.ttc)
    step = filter_step_table(step, args.ttc)

    if pred.empty:
        raise SystemExit("No predictive rows found for the requested target/ttc.")
    if step.empty:
        raise SystemExit("No step aggregate rows found for the requested ttc.")

    plot_family_heatmaps(pred, args.outdir / "fig_temporal_heatmaps.png", args.target, args.ttc)
    plot_entropy_phack_tradeoff(step, args.outdir / "fig_entropy_phack_tradeoff.png", args.ttc)
    plot_temporal_separability_scatter(pred, args.outdir / "fig_temporal_separability.png", args.target, args.ttc)
    write_summary_markdown(pred, step, args.outdir / "README.md", args.target, args.ttc)

    print(f"[DONE] Saved ICML-style ALFWorld temporal figures to {args.outdir}")


if __name__ == "__main__":
    main()
