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
SIGNAL_COLORS = {
    "entropy": "#2f6db3",
    "phack": "#d62728",
    "context": "#2ca02c",
}
RISK_COLOR = "#d62728"
SAFE_COLOR = "#355c7d"
GREY = "#888888"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 10,
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


def ttc_sort_key(label: str):
    if label == "react_baseline":
        return (0, 0)
    m = re.search(r"rtok(\d+)", str(label))
    if m:
        return (1, int(m.group(1)))
    return (2, str(label))


def detect_ttc_regimes(df: pd.DataFrame) -> list[str]:
    if "ttc_label" in df.columns:
        vals = df["ttc_label"].dropna().unique()
    elif "ttc" in df.columns:
        vals = [normalize_ttc_label(x) for x in df["ttc"].dropna().unique()]
    else:
        vals = ["unknown"]
    return sorted({str(v) for v in vals}, key=ttc_sort_key)


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
    for a in ADAPTER_ORDER:
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
        raise ValueError(
            "Cannot derive exploit_action: missing gameable_hack_action/easy_grader_action/fake_completion_action"
        )
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
    adapter: Optional[str],
    ttc: Optional[str],
    method: Optional[str],
    target_col: str,
    target_shift: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    print("[DEBUG] initial rows:", len(df))

    df = ensure_adapter_column(df)

    if "model" in df.columns:
        fam = family.lower()
        df = df[df["model"].astype(str).str.lower().str.contains(fam, regex=False)].copy()
        print("[DEBUG] after family filter:", len(df), "family=", family)

    if (
        method is not None
        and str(method).strip()
        and str(method).lower() not in {"all", "*"}
        and "method" in df.columns
    ):
        df = df[df["method"].astype(str) == str(method)].copy()
        print("[DEBUG] after method filter:", len(df), "method=", method)

    if adapter is not None and str(adapter).lower() not in {"all", "*"} and "adapter" in df.columns:
        a = str(adapter).lower()
        df = df[df["adapter"].astype(str).str.lower().str.contains(a, regex=False)].copy()
        print("[DEBUG] after adapter filter:", len(df), "adapter=", adapter)

    if "ttc" in df.columns:
        df["ttc_label"] = df["ttc"].apply(normalize_ttc_label)
    else:
        if "source_file" in df.columns:
            df["ttc_label"] = df["source_file"].astype(str).map(
                lambda s: "react_baseline" if "react_baseline" in s else normalize_ttc_label(s)
            )
        else:
            df["ttc_label"] = "unknown"

    if ttc is not None and str(ttc).lower() not in {"all", "*"}:
        wanted = normalize_ttc_label(ttc)
        df = df[df["ttc_label"] == wanted].copy()
        print("[DEBUG] after ttc filter:", len(df), "ttc=", ttc, "normalized=", wanted)

    if "source_file" not in df.columns:
        raise ValueError("Expected column 'source_file' in step data.")
    if "step" not in df.columns:
        raise ValueError("Expected column 'step' in step data.")

    df["step_num"] = numeric_series(df["step"])
    df = df.sort_values(["source_file", "step_num"]).reset_index(drop=True)

    ent_cols = find_bin_cols(df, "reasoning_entropy")
    ph_cols = find_bin_cols(df, "reasoning_p")
    if not ent_cols:
        raise ValueError("No columns matching reasoning_entropy_bin0... found.")
    if not ph_cols:
        raise ValueError("No columns matching reasoning_p_bin0... found.")

    df["_target_raw"] = get_target_raw(df, target_col)

    if target_shift == 0:
        df["risk_label"] = df["_target_raw"]
    else:
        df["risk_label"] = df.groupby("source_file", sort=False)["_target_raw"].shift(-target_shift)

    df["risk_label"] = numeric_series(df["risk_label"])
    df = df[df["risk_label"].notna()].copy()

    ent_any = df[ent_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    ph_any = df[ph_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
    df = df[ent_any & ph_any].copy()

    print("[DEBUG] after valid label/trace filter:", len(df))
    print("[DEBUG] TTC regimes:", detect_ttc_regimes(df))
    print("[DEBUG] adapters:", sorted(df["adapter"].astype(str).unique().tolist()))
    return df


def get_action_text(row: pd.Series) -> str:
    for col in ["validated_action", "action", "parsed_action", "raw_action"]:
        if col in row.index and pd.notna(row[col]):
            s = str(row[col])
            if len(s) > 26:
                s = s[:23] + "..."
            return s
    return "action"


def get_curve(row: pd.Series, cols: list[str]) -> np.ndarray:
    y = np.array([pd.to_numeric(row[c], errors="coerce") for c in cols], dtype=float)
    if not np.isfinite(y).any():
        return np.zeros_like(y)
    s = pd.Series(y)
    y = s.interpolate(limit_direction="both").fillna(float(np.nanmean(y))).to_numpy(dtype=float)
    return y


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

    if "step_num" in row.index and "step_num" in df_for_norm.columns:
        step_val = pd.to_numeric(row["step_num"], errors="coerce")
        max_step = pd.to_numeric(df_for_norm["step_num"], errors="coerce").max()
        if pd.notna(step_val) and pd.notna(max_step) and max_step > 0:
            signals.append(float(step_val / max_step))

    base = float(np.mean(signals)) if signals else 0.0
    return np.clip(base + np.linspace(-0.05, 0.05, n_bins), 0, 1)


def choose_episode(df: pd.DataFrame, episode: Optional[str], prefer_risky: bool = True) -> str:
    if episode is not None:
        if episode not in set(df["source_file"].astype(str)):
            raise ValueError(f"Requested episode/source_file not found: {episode}")
        return episode

    agg = (
        df.groupby("source_file")
        .agg(n=("risk_label", "size"), risk=("risk_label", "sum"))
        .reset_index()
    )

    if prefer_risky and (agg["risk"] > 0).any():
        agg = agg[agg["risk"] > 0].copy()
        # Prefer moderately risky but not necessarily fully saturated trajectories
        agg["score"] = (
            2.0 * np.minimum(agg["risk"], 6)
            + 0.25 * np.minimum(agg["n"], 12)
            - 0.15 * np.maximum(agg["risk"] - 6, 0)
        )
        agg = agg.sort_values(["score", "risk", "n"], ascending=False)
    else:
        agg = agg.sort_values(["n", "risk"], ascending=False)

    return str(agg.iloc[0]["source_file"])


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

    # For Mix05 and Mix10 prefer trajectories with some risky transitions but not full saturation.
    if prefer_risky and (agg["risk"] > 0).any():
        if adapter in {"mix05", "mix10"}:
            target = 2
            agg["score"] = -np.abs(agg["risk"] - target) + 0.05 * np.minimum(agg["n"], 10)
            agg = agg[agg["risk"] > 0].copy()
            agg = agg.sort_values(["score", "risk", "n"], ascending=False)
        elif adapter in {"mix50", "hack"}:
            agg["score"] = (
                0.6 * np.minimum(agg["risk"], 4)
                - 0.25 * np.maximum(agg["risk"] - 4, 0)
                + 0.05 * np.minimum(agg["n"], 10)
            )
            agg = agg.sort_values(["score", "risk", "n"], ascending=False)
        else:
            agg["score"] = agg["risk"] + 0.05 * np.minimum(agg["n"], 10)
            agg = agg.sort_values(["score", "risk", "n"], ascending=False)
    else:
        agg = agg.sort_values(["n", "risk"], ascending=False)

    return str(agg.iloc[0]["source_file"])


def select_episode_window(sub: pd.DataFrame, max_steps: int, prefer_risky: bool = True) -> pd.DataFrame:
    sub = sub.sort_values("step_num").reset_index(drop=True)
    if len(sub) <= max_steps:
        return sub

    if prefer_risky and (sub["risk_label"] > 0).any():
        risk_positions = np.where(sub["risk_label"].to_numpy(dtype=float) > 0)[0]
        center = int(risk_positions[0])
        start = max(0, center - max_steps // 2)
        end = min(len(sub), start + max_steps)
        start = max(0, end - max_steps)
        return sub.iloc[start:end].copy()

    return sub.head(max_steps).copy()


def plot_episode_chain(
    df: pd.DataFrame,
    episode_id: str,
    outpath: Path,
    max_steps: int,
    title: str,
) -> None:
    sub = df[df["source_file"].astype(str) == str(episode_id)].copy()
    sub = select_episode_window(sub, max_steps=max_steps, prefer_risky=True)
    if sub.empty:
        raise ValueError(f"No rows for episode {episode_id}")

    ent_cols = find_bin_cols(sub, "reasoning_entropy")
    ph_cols = find_bin_cols(sub, "reasoning_p")
    n_bins = min(len(ent_cols), len(ph_cols))
    ent_cols = ent_cols[:n_bins]
    ph_cols = ph_cols[:n_bins]
    x = np.linspace(0, 1, n_bins)
    n_steps = len(sub)

    fig = plt.figure(figsize=(2.65 * n_steps, 7.4), constrained_layout=True)
    gs = GridSpec(4, n_steps, figure=fig, height_ratios=[1.0, 1.0, 1.0, 0.95])

    row_labels = [
        ("Entropy", SIGNAL_COLORS["entropy"]),
        (r"$p_{\mathrm{hack}}$", SIGNAL_COLORS["phack"]),
        ("Context", SIGNAL_COLORS["context"]),
    ]

    for j, (_, row) in enumerate(sub.iterrows()):
        risk = int(row["risk_label"] > 0)
        action = get_action_text(row)

        curves = [
            get_curve(row, ent_cols),
            get_curve(row, ph_cols),
            get_context_curve(row, n_bins, df),
        ]

        for r in range(3):
            ax = fig.add_subplot(gs[r, j])
            y = curves[r]
            ax.plot(x, y, color=row_labels[r][1], linewidth=2.0)
            ax.fill_between(x, y, alpha=0.10, color=row_labels[r][1])
            ax.axvspan(0.8, 1.0, color="#eeeeee", zorder=0)
            ax.grid(axis="y", alpha=0.18, linewidth=0.6)
            ax.set_xlim(0, 1)

            if r == 0:
                ax.set_title(
                    f"step {int(row['step_num'])}\n"
                    f"{'risky' if risk else 'non-risky'}: {action}",
                    color=RISK_COLOR if risk else "#222222",
                    fontsize=8.5,
                )

            if j == 0:
                ax.set_ylabel(row_labels[r][0])
            else:
                ax.set_yticklabels([])

            if r < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("reasoning\nprogress")

        ax_state = fig.add_subplot(gs[3, j])
        ax_state.axis("off")
        circle_edge = RISK_COLOR if risk else SAFE_COLOR
        circ = patches.Circle((0.15, 0.55), 0.12, facecolor="#f8fbff", edgecolor=circle_edge, linewidth=2.0)
        ax_state.add_patch(circ)
        ax_state.text(0.15, 0.55, f"$s_{{{int(row['step_num'])}}}$", ha="center", va="center", fontsize=13)

        if j < n_steps - 1:
            arrow_color = RISK_COLOR if risk else SAFE_COLOR
            ax_state.annotate(
                "",
                xy=(1.05, 0.55),
                xytext=(0.30, 0.55),
                xycoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="->",
                    color=arrow_color,
                    linewidth=2.2,
                    connectionstyle="arc3,rad=0.22",
                ),
            )
            ax_state.text(0.63, 0.25, action, color=arrow_color, ha="center", va="center", fontsize=8, transform=ax_state.transAxes)

    adapter = sub["adapter"].iloc[0] if "adapter" in sub.columns else "adapter"
    model = sub["model"].iloc[0] if "model" in sub.columns else ""
    ttc = sub["ttc_label"].iloc[0] if "ttc_label" in sub.columns else ""
    n_risk = int(sub["risk_label"].sum())

    fig.suptitle(
        f"{title}\n{model} | {adapter} | {ttc} | risky transitions={n_risk}/{n_steps}",
        fontsize=13,
        y=1.06,
    )
    fig.text(
        0.5,
        0.985,
        "Each column is one reasoning step; transition color marks whether the following action is risky.",
        ha="center",
        va="center",
        fontsize=9.5,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_multi_adapter_chain(
    df: pd.DataFrame,
    outpath: Path,
    max_steps: int,
    title: str,
    adapters: list[str],
) -> None:
    selected = {}
    episode_tables = {}
    ent_cols = find_bin_cols(df, "reasoning_entropy")
    ph_cols = find_bin_cols(df, "reasoning_p")
    n_bins = min(len(ent_cols), len(ph_cols))
    ent_cols = ent_cols[:n_bins]
    ph_cols = ph_cols[:n_bins]
    x = np.linspace(0, 1, n_bins)

    for adapter in adapters:
        ep = choose_episode_for_adapter(df, adapter, prefer_risky=True)
        if ep is None:
            continue
        sub = df[(df["adapter"] == adapter) & (df["source_file"].astype(str) == ep)].copy()
        sub = select_episode_window(sub, max_steps=max_steps, prefer_risky=True)
        if sub.empty:
            continue
        selected[adapter] = ep
        episode_tables[adapter] = sub

    if not episode_tables:
        raise RuntimeError("No adapter-specific episodes available for multi-adapter plot.")

    n_adapters = len(episode_tables)
    n_cols = max(len(sub) for sub in episode_tables.values())
    n_rows = 4 * n_adapters

    fig = plt.figure(figsize=(2.55 * n_cols, 1.85 * n_rows), constrained_layout=True)
    gs = GridSpec(n_rows, n_cols, figure=fig)

    adapter_list = [a for a in adapters if a in episode_tables]

    for a_idx, adapter in enumerate(adapter_list):
        sub = episode_tables[adapter]
        row0 = 4 * a_idx
        adapter_color = ADAPTER_COLORS.get(adapter, "#444444")

        # Adapter label on the far left
        for local_step_idx in range(n_cols):
            if local_step_idx >= len(sub):
                for rr in range(4):
                    ax_blank = fig.add_subplot(gs[row0 + rr, local_step_idx])
                    ax_blank.axis("off")
                continue

            row = sub.iloc[local_step_idx]
            risk = int(row["risk_label"] > 0)
            action = get_action_text(row)

            curves = [
                get_curve(row, ent_cols),
                get_curve(row, ph_cols),
                get_context_curve(row, n_bins, df),
            ]

            row_specs = [
                ("Entropy", SIGNAL_COLORS["entropy"]),
                (r"$p_{\mathrm{hack}}$", SIGNAL_COLORS["phack"]),
                ("Context", SIGNAL_COLORS["context"]),
            ]

            for rr in range(3):
                ax = fig.add_subplot(gs[row0 + rr, local_step_idx])
                y = curves[rr]
                ax.plot(x, y, color=row_specs[rr][1], linewidth=2.0)
                ax.fill_between(x, y, alpha=0.10, color=row_specs[rr][1])
                ax.axvspan(0.8, 1.0, color="#eeeeee", zorder=0)
                ax.grid(axis="y", alpha=0.18, linewidth=0.6)
                ax.set_xlim(0, 1)

                if rr == 0:
                    ax.set_title(
                        f"step {int(row['step_num'])}\n"
                        f"{'risky' if risk else 'non-risky'}: {action}",
                        color=RISK_COLOR if risk else "#222222",
                        fontsize=8.0,
                    )

                if local_step_idx == 0:
                    ax.set_ylabel(f"{ADAPTER_LABELS.get(adapter, adapter)}\n{row_specs[rr][0]}", color=adapter_color)
                else:
                    ax.set_yticklabels([])

                if rr < 2:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("reasoning\nprogress")

                # Adapter block border effect
                for spine in ax.spines.values():
                    spine.set_linewidth(1.0)
                ax.spines["left"].set_color(adapter_color)

            ax_state = fig.add_subplot(gs[row0 + 3, local_step_idx])
            ax_state.axis("off")
            circle_edge = RISK_COLOR if risk else SAFE_COLOR
            circ = patches.Circle((0.15, 0.55), 0.12, facecolor="#f8fbff", edgecolor=circle_edge, linewidth=2.0)
            ax_state.add_patch(circ)
            ax_state.text(0.15, 0.55, f"$s_{{{int(row['step_num'])}}}$", ha="center", va="center", fontsize=12)

            if local_step_idx < len(sub) - 1:
                arrow_color = RISK_COLOR if risk else SAFE_COLOR
                ax_state.annotate(
                    "",
                    xy=(1.05, 0.55),
                    xytext=(0.30, 0.55),
                    xycoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="->",
                        color=arrow_color,
                        linewidth=2.0,
                        connectionstyle="arc3,rad=0.22",
                    ),
                )
                ax_state.text(
                    0.63, 0.23, action,
                    color=arrow_color,
                    ha="center", va="center",
                    fontsize=7.5,
                    transform=ax_state.transAxes,
                )

            if local_step_idx == 0:
                ax_state.text(
                    -0.02, 0.55,
                    ADAPTER_LABELS.get(adapter, adapter),
                    ha="right", va="center",
                    fontsize=11, fontweight="bold",
                    color=adapter_color,
                    transform=ax_state.transAxes,
                )

    family = df["model"].iloc[0] if "model" in df.columns and not df.empty else ""
    ttc = df["ttc_label"].iloc[0] if "ttc_label" in df.columns and not df.empty else ""
    fig.suptitle(
        f"{title}\n{family} | {ttc} | one representative episode per adapter",
        fontsize=14,
        y=1.01,
    )
    fig.text(
        0.5,
        0.992,
        "Rows = adapters. Within each adapter block: entropy, reward-hack activation, context, and state transition.",
        ha="center",
        va="center",
        fontsize=9.5,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_concatenated(
    df: pd.DataFrame,
    outpath: Path,
    signal: str,
    max_rows: Optional[int] = None,
) -> None:
    if signal == "phack":
        cols = find_bin_cols(df, "reasoning_p")
        ylabel = r"Reasoning $p_{\mathrm{hack}}$"
    elif signal == "entropy":
        cols = find_bin_cols(df, "reasoning_entropy")
        ylabel = "Reasoning entropy"
    else:
        raise ValueError("signal must be 'phack' or 'entropy'")

    adapters = [a for a in ADAPTER_ORDER if a in set(df["adapter"].astype(str))]
    n_adapters = len(adapters)
    fig, axes = plt.subplots(n_adapters, 1, figsize=(11, 2.7 * n_adapters), sharex=False, constrained_layout=True)
    if n_adapters == 1:
        axes = [axes]

    for ax, adapter in zip(axes, adapters):
        sub = df[df["adapter"] == adapter].sort_values(["source_file", "step_num"]).copy()
        if max_rows is not None:
            sub = sub.head(max_rows)

        xs, ys = [], []
        boundaries = []
        cursor = 0

        for _, row in sub.iterrows():
            y = get_curve(row, cols)
            n = len(y)
            xs.extend(list(range(cursor, cursor + n)))
            ys.extend(y.tolist())
            risk = int(row["risk_label"] > 0)
            boundaries.append((cursor + n - 0.5, risk))
            cursor += n

        color = ADAPTER_COLORS.get(adapter, "#444444")
        ax.plot(xs, ys, color=color, linewidth=1.2)
        for xline, risk in boundaries:
            ax.axvline(
                xline,
                color=RISK_COLOR if risk else GREY,
                linestyle="--",
                linewidth=0.8,
                alpha=0.75,
            )
        ax.set_ylabel(f"{ADAPTER_LABELS.get(adapter, adapter)}\n{ylabel}")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_title(f"Concatenated raw {signal} trajectories with action boundaries")
    axes[-1].set_xlabel("Concatenated reasoning-token bins")
    axes[0].text(
        0.01,
        0.98,
        "Red dashed = risky following action; gray dashed = non-risky following action",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, outpath: Path) -> None:
    summary = (
        df.groupby(["model", "adapter", "ttc_label"], dropna=False)
        .agg(
            n=("risk_label", "size"),
            n_risky=("risk_label", "sum"),
            mean_risk=("risk_label", "mean"),
        )
        .reset_index()
        .sort_values(["model", "adapter", "ttc_label"], key=lambda s: s.map(str))
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outpath.with_suffix(".csv"), index=False)

    lines = ["# Markov-chain temporal visualization summary\n"]
    lines.append("This file summarizes valid rows and risky following actions.\n")
    try:
        lines.append(summary.to_markdown(index=False))
    except Exception:
        lines.append(summary.to_string(index=False))
    outpath.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps-csv",
        type=Path,
        default=Path("BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("figures/markov_temporal"))
    parser.add_argument("--family", type=str, default="Qwen", choices=["Qwen", "Falcon", "Llama"])
    parser.add_argument("--adapter", type=str, default="mix50", choices=ADAPTER_ORDER + ["all"])
    parser.add_argument("--ttc", type=str, default="all", help="e.g. ttc_rtok32, 32, react_baseline, all")
    parser.add_argument("--method", type=str, default="all", help="method filter or all")
    parser.add_argument("--target-col", type=str, default="bad_action")
    parser.add_argument(
        "--target-shift",
        type=int,
        default=0,
        help="0 = current action after reasoning trace; 1 = next decision step action",
    )
    parser.add_argument("--episode", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-concat-rows", type=int, default=120)
    parser.add_argument("--prefer-risky", action="store_true")
    args = parser.parse_args()

    configure_style()

    adapter = None if args.adapter == "all" else args.adapter
    df = load_steps(
        path=args.steps_csv,
        family=args.family,
        adapter=adapter,
        ttc=args.ttc,
        method=args.method,
        target_col=args.target_col,
        target_shift=args.target_shift,
    )

    if df.empty:
        raise RuntimeError("No data after filtering. Check family/adapter/TTC/method filters.")

    ttc_regimes = detect_ttc_regimes(df)
    print("[INFO] TTC regimes found:", ttc_regimes)

    for regime in ttc_regimes:
        df_reg = df[df["ttc_label"] == regime].copy()
        if df_reg.empty:
            continue

        stem = f"{args.family.lower()}_{args.adapter}_{regime}_{args.target_col}_shift{args.target_shift}"
        phack_out = args.outdir / f"concat_phack_{stem}.png"
        entropy_out = args.outdir / f"concat_entropy_{stem}.png"
        summary_out = args.outdir / f"summary_{stem}.md"

        if args.adapter == "all":
            multi_out = args.outdir / f"markov_chain_multi_adapter_{stem}.png"
            print(f"[INFO] Plotting multi-adapter regime={regime}, rows={len(df_reg)}")
            plot_multi_adapter_chain(
                df=df_reg,
                outpath=multi_out,
                max_steps=args.max_steps,
                title=f"Decision-local monitoring as a Markov chain ({regime})",
                adapters=[a for a in ADAPTER_ORDER if a in set(df_reg['adapter'].astype(str))],
            )
            print("[DONE] Generated:")
            print(" ", multi_out)
            print(" ", multi_out.with_suffix(".pdf"))
        else:
            try:
                episode = choose_episode(
                    df_reg,
                    args.episode,
                    prefer_risky=bool(args.prefer_risky),
                )
            except Exception as e:
                print(f"[WARN] Could not choose episode for {regime}: {e}")
                continue

            chain_out = args.outdir / f"markov_chain_{stem}.png"
            print(f"[INFO] Plotting regime={regime}, rows={len(df_reg)}, episode={episode}")
            plot_episode_chain(
                df=df_reg,
                episode_id=episode,
                outpath=chain_out,
                max_steps=args.max_steps,
                title=f"Decision-local monitoring as a Markov chain ({regime})",
            )
            print("[DONE] Generated:")
            print(" ", chain_out)
            print(" ", chain_out.with_suffix(".pdf"))

        plot_concatenated(
            df=df_reg,
            outpath=phack_out,
            signal="phack",
            max_rows=args.max_concat_rows,
        )
        plot_concatenated(
            df=df_reg,
            outpath=entropy_out,
            signal="entropy",
            max_rows=args.max_concat_rows,
        )
        write_summary(df_reg, summary_out)

        print(" ", phack_out)
        print(" ", phack_out.with_suffix(".pdf"))
        print(" ", entropy_out)
        print(" ", entropy_out.with_suffix(".pdf"))
        print(" ", summary_out)
        print(" ", summary_out.with_suffix(".csv"))


if __name__ == "__main__":
    main()