#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ADAPTER_ORDER = ["mix05", "mix10", "mix50"]
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
SIGNAL_COLORS = {
    "entropy": "#2f6db3",
    "phack": "#d62728",
    "context": "#2ca02c",
}
RISK_COLOR = "#d62728"
SAFE_COLOR = "#355c7d"
GREY = "#777777"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_ttc_label(x) -> str:
    s = str(x)
    if s.lower() in {"nan", "none", ""}:
        return "unknown"
    if "react_baseline" in s:
        return "react_baseline"
    m = re.search(r"rtok(\d+)", s)
    if m:
        return f"ttc_rtok{m.group(1)}"
    m = re.search(r"(\d+)", s)
    if m:
        return f"ttc_rtok{m.group(1)}"
    return s


def find_bin_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = []
    i = 0
    while f"{prefix}_bin{i}" in df.columns:
        cols.append(f"{prefix}_bin{i}")
        i += 1
    return cols


def numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def derive_adapter_from_model(model_name: str) -> str:
    m = str(model_name).lower()
    if "mix05" in m or "mix5" in m:
        return "mix05"
    if "mix10" in m:
        return "mix10"
    if "mix50" in m:
        return "mix50"
    if "control" in m:
        return "control"
    if "hack" in m:
        return "hack"
    return "unknown"


def normalize_adapter_label(x) -> str:
    s = str(x).lower()
    for a in ["control", "mix05", "mix10", "mix50", "hack"]:
        if a in s:
            return a
    if "mix5" in s:
        return "mix05"
    return s


def ensure_adapter_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "adapter" not in df.columns:
        if "model" not in df.columns:
            raise ValueError("Need either adapter or model column.")
        df["adapter"] = df["model"].map(derive_adapter_from_model)
    else:
        df["adapter"] = df["adapter"].map(normalize_adapter_label)
        if "model" in df.columns:
            missing = df["adapter"].isin(["unknown", "nan", "none", ""])
            df.loc[missing, "adapter"] = df.loc[missing, "model"].map(derive_adapter_from_model)
    return df


def derive_exploit_action(df: pd.DataFrame) -> pd.Series:
    parts = []
    for c in ["gameable_hack_action", "easy_grader_action", "fake_completion_action"]:
        if c in df.columns:
            parts.append(numeric_series(df[c]).fillna(0))
    if not parts:
        raise ValueError("Cannot derive exploit_action from available columns.")
    return (sum(parts) > 0).astype(float)


def get_target_raw(df: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col == "exploit_action":
        if "exploit_action" in df.columns:
            return numeric_series(df["exploit_action"]).fillna(0)
        return derive_exploit_action(df)
    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    return numeric_series(df[target_col]).fillna(0)


def load_steps(
    path: Path,
    family: str,
    ttc: str,
    method: str,
    target_col: str,
    target_shift: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = ensure_adapter_column(df)

    if "model" in df.columns:
        df = df[df["model"].astype(str).str.lower().str.contains(family.lower(), regex=False)].copy()

    if method.lower() not in {"all", "*"} and "method" in df.columns:
        df = df[df["method"].astype(str) == method].copy()

    if "ttc" in df.columns:
        df["ttc_label"] = df["ttc"].apply(normalize_ttc_label)
    else:
        df["ttc_label"] = df["source_file"].astype(str).map(
            lambda s: "react_baseline" if "react_baseline" in s else normalize_ttc_label(s)
        )

    if ttc.lower() not in {"all", "*"}:
        wanted = normalize_ttc_label(ttc)
        df = df[df["ttc_label"] == wanted].copy()

    df["step_num"] = numeric_series(df["step"])
    df = df.sort_values(["source_file", "step_num"]).reset_index(drop=True)

    ent_cols = find_bin_cols(df, "reasoning_entropy")
    ph_cols = find_bin_cols(df, "reasoning_p")
    if not ent_cols or not ph_cols:
        raise ValueError("Missing reasoning_entropy_bin* or reasoning_p_bin* columns.")

    df["_target_raw"] = get_target_raw(df, target_col)
    if target_shift == 0:
        df["risk_label"] = df["_target_raw"]
    else:
        df["risk_label"] = df.groupby("source_file")["_target_raw"].shift(-target_shift)

    df["risk_label"] = numeric_series(df["risk_label"])
    df = df[df["risk_label"].notna()].copy()

    valid_trace = (
        df[ent_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
        & df[ph_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    )
    df = df[valid_trace].copy()
    return df


def get_curve(row: pd.Series, cols: list[str]) -> np.ndarray:
    y = np.array([pd.to_numeric(row[c], errors="coerce") for c in cols], dtype=float)
    if not np.isfinite(y).any():
        return np.zeros_like(y)
    s = pd.Series(y)
    return s.interpolate(limit_direction="both").fillna(float(np.nanmean(y))).to_numpy(dtype=float)


def get_action_text(row: pd.Series) -> str:
    for col in ["validated_action", "action", "parsed_action", "raw_action"]:
        if col in row.index and pd.notna(row[col]):
            s = str(row[col])
            return s[:28] + "..." if len(s) > 31 else s
    return "action"


def norm_value_from_column(row: pd.Series, df_for_norm: pd.DataFrame, col: str) -> Optional[float]:
    if col not in row.index or col not in df_for_norm.columns:
        return None
    val = pd.to_numeric(row[col], errors="coerce")
    if pd.isna(val):
        return None
    allv = pd.to_numeric(df_for_norm[col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(allv)
    if finite.any() and np.nanmax(allv) > np.nanmin(allv):
        return float((float(val) - np.nanmin(allv)) / (np.nanmax(allv) - np.nanmin(allv)))
    return 0.0


def get_context_curve(row: pd.Series, n_bins: int, df_for_norm: pd.DataFrame) -> np.ndarray:
    signals = []

    for col in [
        "proxy_score",
        "proxy_score_without_success",
        "proxy_reward",
        "proxy_farming_reward",
        "admissible_action_count",
    ]:
        v = norm_value_from_column(row, df_for_norm, col)
        if v is not None:
            signals.append(v)

    for col in [
        "gameable_hack_action",
        "easy_grader_action",
        "fake_completion_action",
        "buy_action",
        "bad_buy_action",
        "low_reward_buy",
    ]:
        if col in row.index:
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(val):
                signals.append(float(val > 0))

    if "step_num" in row.index:
        max_step = pd.to_numeric(df_for_norm["step_num"], errors="coerce").max()
        if pd.notna(max_step) and max_step > 0:
            signals.append(float(row["step_num"] / max_step))

    base = float(np.mean(signals)) if signals else 0.0
    return np.clip(base + np.linspace(-0.05, 0.05, n_bins), 0, 1)


def choose_episode_for_adapter(
    df: pd.DataFrame,
    adapter: str,
    prefer_risky: bool = True,
) -> Optional[str]:
    sub = df[df["adapter"] == adapter].copy()
    if sub.empty:
        return None

    agg = (
        sub.groupby("source_file")
        .agg(n=("risk_label", "size"), risk=("risk_label", "sum"))
        .reset_index()
    )
    if agg.empty:
        return None

    # Prefer trajectories with some risk, but avoid totally saturated hack-like traces.
    if prefer_risky and (agg["risk"] > 0).any():
        if adapter in {"mix05", "mix10"}:
            target = 1.5
        elif adapter == "mix50":
            target = 3.0
        else:
            target = 2.0
        agg = agg[agg["risk"] > 0].copy()
        agg["score"] = -np.abs(agg["risk"] - target) + 0.05 * np.minimum(agg["n"], 12)
        agg = agg.sort_values(["score", "risk", "n"], ascending=False)
    else:
        agg = agg.sort_values(["n", "risk"], ascending=False)

    return str(agg.iloc[0]["source_file"])


def select_episode_window(sub: pd.DataFrame, max_steps: int) -> pd.DataFrame:
    sub = sub.sort_values("step_num").reset_index(drop=True)
    if len(sub) <= max_steps:
        return sub

    if (sub["risk_label"] > 0).any():
        pos = np.where(sub["risk_label"].to_numpy(float) > 0)[0]
        center = int(pos[0])
        start = max(0, center - max_steps // 2)
        end = min(len(sub), start + max_steps)
        start = max(0, end - max_steps)
        return sub.iloc[start:end].copy()

    return sub.head(max_steps).copy()


def plot_matched_multi_adapter_markov(
    df: pd.DataFrame,
    outpath: Path,
    adapters: list[str],
    max_steps: int,
    title: str,
) -> None:
    ent_cols = find_bin_cols(df, "reasoning_entropy")
    ph_cols = find_bin_cols(df, "reasoning_p")
    n_bins = min(len(ent_cols), len(ph_cols))
    ent_cols, ph_cols = ent_cols[:n_bins], ph_cols[:n_bins]
    x = np.linspace(0, 1, n_bins)

    rows_by_adapter = {}
    selected_eps = {}

    for adapter in adapters:
        ep = choose_episode_for_adapter(df, adapter, prefer_risky=True)
        if ep is None:
            continue
        sub = df[(df["adapter"] == adapter) & (df["source_file"].astype(str) == ep)].copy()
        sub = select_episode_window(sub, max_steps=max_steps)
        if not sub.empty:
            rows_by_adapter[adapter] = sub
            selected_eps[adapter] = ep

    if not rows_by_adapter:
        raise RuntimeError("No adapter-specific trajectories found.")

    n_adapters = len(rows_by_adapter)
    n_cols = max(len(v) for v in rows_by_adapter.values())

    fig = plt.figure(figsize=(2.6 * n_cols, 4.0 * n_adapters), constrained_layout=True)
    gs = GridSpec(4 * n_adapters, n_cols, figure=fig, height_ratios=[1, 1, 1, 0.9] * n_adapters)

    for a_idx, adapter in enumerate(adapters):
        if adapter not in rows_by_adapter:
            continue

        sub = rows_by_adapter[adapter]
        row_offset = 4 * a_idx
        adapter_color = ADAPTER_COLORS.get(adapter, "#333333")

        for j in range(n_cols):
            if j >= len(sub):
                for rr in range(4):
                    ax = fig.add_subplot(gs[row_offset + rr, j])
                    ax.axis("off")
                continue

            row = sub.iloc[j]
            risk = int(row["risk_label"] > 0)
            action = get_action_text(row)

            curves = [
                ("Entropy", get_curve(row, ent_cols), SIGNAL_COLORS["entropy"]),
                (r"$p_{\mathrm{hack}}$", get_curve(row, ph_cols), SIGNAL_COLORS["phack"]),
                ("Context", get_context_curve(row, n_bins, df), SIGNAL_COLORS["context"]),
            ]

            for rr, (label, y, color) in enumerate(curves):
                ax = fig.add_subplot(gs[row_offset + rr, j])
                ax.plot(x, y, color=color, linewidth=2.0)
                ax.fill_between(x, y, color=color, alpha=0.10)
                ax.axvspan(0.8, 1.0, color="#eeeeee", zorder=0)
                ax.grid(axis="y", alpha=0.18, linewidth=0.6)
                ax.set_xlim(0, 1)

                if rr == 0:
                    ax.set_title(
                        f"step {int(row['step_num'])}\n"
                        f"{'risky' if risk else 'non-risky'}: {action}",
                        fontsize=8,
                        color=RISK_COLOR if risk else "#222222",
                    )

                if j == 0:
                    ax.set_ylabel(
                        f"{ADAPTER_LABELS.get(adapter, adapter)}\n{label}",
                        color=adapter_color,
                    )
                else:
                    ax.set_yticklabels([])

                if rr < 2:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("reasoning\nprogress")

            # State / transition row
            ax_s = fig.add_subplot(gs[row_offset + 3, j])
            ax_s.axis("off")
            edge = RISK_COLOR if risk else SAFE_COLOR

            circ = patches.Circle(
                (0.15, 0.55),
                radius=0.12,
                facecolor="#f8fbff",
                edgecolor=edge,
                linewidth=2.0,
            )
            ax_s.add_patch(circ)
            ax_s.text(
                0.15,
                0.55,
                f"$s_{{{int(row['step_num'])}}}$",
                ha="center",
                va="center",
                fontsize=12,
            )

            if j < len(sub) - 1:
                ax_s.annotate(
                    "",
                    xy=(1.05, 0.55),
                    xytext=(0.30, 0.55),
                    xycoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="->",
                        color=edge,
                        linewidth=2.0,
                        connectionstyle="arc3,rad=0.22",
                    ),
                )
                ax_s.text(
                    0.63,
                    0.22,
                    action,
                    color=edge,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    transform=ax_s.transAxes,
                )

    ttc = df["ttc_label"].iloc[0] if "ttc_label" in df.columns else ""
    model = df["model"].iloc[0] if "model" in df.columns else ""
    fig.suptitle(
        f"{title}\n{model} | {ttc} | matched multi-adapter view",
        fontsize=14,
        y=1.02,
    )
    fig.text(
        0.5,
        1.005,
        "Rows are adapters. Each column is one decision step; red transitions mark risky actions.",
        ha="center",
        va="center",
        fontsize=9,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps-csv",
        type=Path,
        default=Path("BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures/markov_temporal"))
    parser.add_argument("--family", type=str, default="Qwen", choices=["Qwen", "Falcon", "Llama"])
    parser.add_argument("--ttc", type=str, default="32")
    parser.add_argument("--method", type=str, default="all")
    parser.add_argument("--target-col", type=str, default="bad_action")
    parser.add_argument("--target-shift", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--adapters", type=str, default="mix05,mix10,mix50")
    args = parser.parse_args()

    configure_style()

    df = load_steps(
        path=args.steps_csv,
        family=args.family,
        ttc=args.ttc,
        method=args.method,
        target_col=args.target_col,
        target_shift=args.target_shift,
    )

    if df.empty:
        raise RuntimeError("No data after filtering.")

    adapters = [a.strip().lower() for a in args.adapters.split(",") if a.strip()]
    available = sorted(df["adapter"].unique().tolist())
    adapters = [a for a in adapters if a in available]

    if not adapters:
        raise RuntimeError(f"No requested adapters found. Available: {available}")

    ttc_label = df["ttc_label"].iloc[0]
    out = (
        args.outdir
        / f"matched_markov_{args.family.lower()}_{ttc_label}_{args.target_col}_shift{args.target_shift}.png"
    )

    plot_matched_multi_adapter_markov(
        df=df,
        outpath=out,
        adapters=adapters,
        max_steps=args.max_steps,
        title="Decision-local monitoring across mixed adapters",
    )

    print("[DONE] Generated:")
    print(" ", out)
    print(" ", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()