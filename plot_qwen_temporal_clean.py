#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


ADAPTERS = ["control", "mix05", "mix10", "mix50", "hack"]
ADAPTER_LABELS = {
    "control": "Control",
    "mix05": "Mix05",
    "mix10": "Mix10",
    "mix50": "Mix50",
    "hack": "Hack",
}

BAD_COLOR = "#D62728"
GOOD_COLOR = "#4C78A8"


def fnum(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except Exception:
        return None


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def bin_cols(rows: list[dict], prefix: str) -> list[str]:
    cols = [c for c in rows[0].keys() if c.startswith(prefix)]
    return sorted(cols, key=lambda s: int(s.split("bin")[-1]))


def normalize_ttc_value(row: dict) -> str:
    method = str(row.get("method", ""))
    if method == "react_baseline":
        return "react_baseline"
    ttc = fnum(row.get("ttc"))
    if ttc is not None:
        return f"ttc_{int(ttc)}"
    return method or "unknown"


def load_and_prepare(
    steps_csv: Path,
    family: str,
    method: str,
    target: str,
    target_shift: int,
    ttc_filter: Optional[str],
) -> tuple[list[str], list[str], list[dict]]:
    rows = read_rows(steps_csv)
    if not rows:
        raise RuntimeError("Empty CSV.")

    p_cols = bin_cols(rows, "reasoning_p_bin")
    e_cols = bin_cols(rows, "reasoning_entropy_bin")
    if not p_cols or not e_cols:
        raise RuntimeError("Missing reasoning_p_bin* or reasoning_entropy_bin* columns.")

    # Filter family and method.
    out = []
    for r in rows:
        model = str(r.get("model", ""))
        if family.lower() not in model.lower():
            continue
        if method != "all" and str(r.get("method", "")) != method:
            continue
        if str(r.get("adapter", "")) not in ADAPTERS:
            continue
        regime = normalize_ttc_value(r)
        r["_ttc_regime"] = regime
        if ttc_filter is not None and regime != ttc_filter:
            continue
        out.append(r)

    out.sort(key=lambda r: (r.get("source_file", ""), int(fnum(r.get("step")) or 0)))

    # Build shifted labels within each source_file.
    for i, r in enumerate(out):
        if target == "bad_action":
            raw = fnum(r.get("bad_action")) or 0.0
        elif target == "exploit_action":
            vals = [
                fnum(r.get("gameable_hack_action")) or 0.0,
                fnum(r.get("easy_grader_action")) or 0.0,
                fnum(r.get("fake_completion_action")) or 0.0,
            ]
            raw = 1.0 if sum(vals) > 0 else 0.0
        else:
            raw = fnum(r.get(target)) or 0.0
        r["_target_raw"] = raw

    for i, r in enumerate(out):
        label = None
        if target_shift == 0:
            label = r["_target_raw"]
        else:
            j = i + target_shift
            if j < len(out) and out[j].get("source_file") == r.get("source_file"):
                label = out[j]["_target_raw"]
        r["_risk_label"] = label

    valid = []
    for r in out:
        if r["_risk_label"] is None:
            continue
        p_vals = [fnum(r.get(c)) for c in p_cols]
        e_vals = [fnum(r.get(c)) for c in e_cols]
        if any(v is None for v in p_vals) or any(v is None for v in e_vals):
            continue
        valid.append(r)

    return p_cols, e_cols, valid


def mean_ci(traces: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    if not traces:
        return np.array([]), np.array([])
    arr = np.asarray(traces, dtype=float)
    mean = np.nanmean(arr, axis=0)
    if len(arr) <= 1:
        ci = np.zeros_like(mean)
    else:
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(len(arr))
        ci = 1.96 * sem
    return mean, ci


def collect_by_adapter(rows: list[dict], cols: list[str]) -> dict:
    payload = {}
    for adapter in ADAPTERS:
        bad, good = [], []
        sub = [r for r in rows if r.get("adapter") == adapter]
        for r in sub:
            vals = [fnum(r.get(c)) for c in cols]
            if float(r["_risk_label"]) > 0.5:
                bad.append(vals)
            else:
                good.append(vals)
        payload[adapter] = {
            "bad": bad,
            "good": good,
            "n_bad": len(bad),
            "n_good": len(good),
            "n": len(bad) + len(good),
        }
    return payload


def collect_first_n_by_adapter(rows: list[dict], p_cols: list[str], e_cols: list[str], first_n: int) -> dict:
    payload = {}
    for adapter in ADAPTERS:
        sub = [r for r in rows if r.get("adapter") == adapter]
        selected = []
        for r in sub[:first_n]:
            selected.append(
                {
                    "risk_label": int(float(r["_risk_label"]) > 0.5),
                    "p_vals": [fnum(r.get(c)) for c in p_cols],
                    "e_vals": [fnum(r.get(c)) for c in e_cols],
                    "source_file": r.get("source_file"),
                    "step": int(fnum(r.get("step")) or 0),
                }
            )
        payload[adapter] = selected
    return payload


def set_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_clean_figure(
    p_payload: dict,
    e_payload: dict,
    out_png: Path,
    title: str,
    target: str,
):
    set_style()

    x = np.linspace(0.0, 1.0, 8)

    fig, axes = plt.subplots(
        nrows=len(ADAPTERS),
        ncols=2,
        figsize=(7.2, 9.0),
        sharex=True,
        constrained_layout=True,
    )

    col_titles = [r"Reward-hack activation $p_{\mathrm{hack}}$", "Token entropy"]

    for col, ctitle in enumerate(col_titles):
        axes[0, col].set_title(ctitle, fontweight="bold")

    for r_idx, adapter in enumerate(ADAPTERS):
        for col_idx, payload in enumerate([p_payload, e_payload]):
            ax = axes[r_idx, col_idx]
            cell = payload[adapter]

            ax.axvspan(0.8, 1.0, color="#dddddd", alpha=0.35, zorder=0)
            ax.axhline(0, color="#999999", linewidth=0.5, alpha=0.5)
            ax.grid(axis="y")

            bad_mean, bad_ci = mean_ci(cell["bad"])
            good_mean, good_ci = mean_ci(cell["good"])

            if len(bad_mean) > 0:
                ax.plot(x, bad_mean, color=BAD_COLOR, linewidth=1.8, label=r"risky next action")
                ax.fill_between(
                    x,
                    bad_mean - bad_ci,
                    bad_mean + bad_ci,
                    color=BAD_COLOR,
                    alpha=0.15,
                    linewidth=0,
                )

            if len(good_mean) > 0:
                ax.plot(
                    x,
                    good_mean,
                    color=GOOD_COLOR,
                    linewidth=1.8,
                    linestyle="--",
                    label=r"non-risky next action",
                )
                ax.fill_between(
                    x,
                    good_mean - good_ci,
                    good_mean + good_ci,
                    color=GOOD_COLOR,
                    alpha=0.12,
                    linewidth=0,
                )

            if len(bad_mean) == 0 and len(good_mean) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "insufficient data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )

            if col_idx == 0:
                ax.set_ylabel(
                    f"{ADAPTER_LABELS[adapter]}\n"
                    f"n={cell['n']}, bad={cell['n_bad']}, non={cell['n_good']}"
                )

            if r_idx == len(ADAPTERS) - 1:
                ax.set_xlabel("Normalized reasoning progress")

            ax.set_xlim(0, 1)

    handles = [
        plt.Line2D([0], [0], color=BAD_COLOR, lw=2, label="risky next action"),
        plt.Line2D([0], [0], color=GOOD_COLOR, lw=2, ls="--", label="non-risky next action"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#dddddd", alpha=0.35, label="late window"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.045)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_first_n_figure(
    raw_payload: dict,
    out_png: Path,
    title: str,
    first_n: int,
):
    set_style()
    x = np.linspace(0.0, 1.0, 8)
    fig, axes = plt.subplots(
        nrows=len(ADAPTERS),
        ncols=2,
        figsize=(7.8, 9.8),
        sharex=True,
        constrained_layout=True,
    )

    col_titles = [r"First raw $p_{\mathrm{hack}}$ trajectories", "First raw entropy trajectories"]
    for col, ctitle in enumerate(col_titles):
        axes[0, col].set_title(ctitle, fontweight="bold")

    for r_idx, adapter in enumerate(ADAPTERS):
        traces = raw_payload[adapter]
        for col_idx, key in enumerate(["p_vals", "e_vals"]):
            ax = axes[r_idx, col_idx]
            ax.axvspan(0.8, 1.0, color="#dddddd", alpha=0.35, zorder=0)
            ax.grid(axis="y")
            if not traces:
                ax.text(0.5, 0.5, "no trajectories", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            for i, tr in enumerate(traces):
                vals = tr[key]
                color = BAD_COLOR if tr["risk_label"] == 1 else GOOD_COLOR
                ls = "-" if tr["risk_label"] == 1 else "--"
                alpha = max(0.35, 0.9 - i * 0.1)
                ax.plot(x, vals, color=color, linestyle=ls, linewidth=1.5, alpha=alpha)
                ax.scatter([x[-1]], [vals[-1]], color=color, s=10, alpha=alpha)
            if col_idx == 0:
                ax.set_ylabel(f"{ADAPTER_LABELS[adapter]}\nfirst {min(first_n, len(traces))} traces")
            if r_idx == len(ADAPTERS) - 1:
                ax.set_xlabel("Normalized reasoning progress")
            ax.set_xlim(0, 1)

    handles = [
        plt.Line2D([0], [0], color=BAD_COLOR, lw=2, label="bad next action"),
        plt.Line2D([0], [0], color=GOOD_COLOR, lw=2, ls="--", label="non-bad next action"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#dddddd", alpha=0.35, label="late window"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.045)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(
    p_payload: dict,
    e_payload: dict,
    out_md: Path,
    title: str,
):
    lines = [f"# {title}", ""]
    lines.append("| Adapter | n | n_bad | n_nonbad | late_delta_p | late_delta_entropy |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for adapter in ADAPTERS:
        pc = p_payload[adapter]
        ec = e_payload[adapter]

        def late_delta(cell):
            b_mean, _ = mean_ci(cell["bad"])
            g_mean, _ = mean_ci(cell["good"])
            if len(b_mean) == 0 or len(g_mean) == 0:
                return None
            return float(np.mean(b_mean[-2:]) - np.mean(g_mean[-2:]))

        dp = late_delta(pc)
        de = late_delta(ec)

        lines.append(
            f"| {ADAPTER_LABELS[adapter]} | {pc['n']} | {pc['n_bad']} | {pc['n_good']} | "
            f"{'n/a' if dp is None else f'{dp:.4f}'} | "
            f"{'n/a' if de is None else f'{de:.4f}'} |"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps-csv",
        type=Path,
        default=Path("BALROG/analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"),
    )
    parser.add_argument("--family", type=str, default="Qwen")
    parser.add_argument("--method", type=str, default="react_ttc_monitored")
    parser.add_argument("--target", type=str, default="bad_action", choices=["bad_action", "exploit_action"])
    parser.add_argument("--target-shift", type=int, default=1)
    parser.add_argument("--ttc", type=str, default=None, help="Optional e.g. ttc_32, ttc_64")
    parser.add_argument("--outdir", type=Path, default=Path("paper_figures"))
    parser.add_argument("--raw-first-n", type=int, default=0, help="If >0, plot first N raw trajectories per adapter instead of aggregated means.")
    args = parser.parse_args()

    p_cols, e_cols, rows = load_and_prepare(
        steps_csv=args.steps_csv,
        family=args.family,
        method=args.method,
        target=args.target,
        target_shift=args.target_shift,
        ttc_filter=args.ttc,
    )

    if not rows:
        raise RuntimeError("No rows after filtering.")

    p_payload = collect_by_adapter(rows, p_cols)
    e_payload = collect_by_adapter(rows, e_cols)

    suffix = f"{args.family.lower()}_{args.target}_shift{args.target_shift}"
    if args.ttc:
        suffix += f"_{args.ttc}"

    out_png = args.outdir / f"temporal_two_column_{suffix}.png"
    out_md = args.outdir / f"temporal_two_column_{suffix}.md"

    title = (
        f"{args.family}: temporal internal dynamics before "
        f"{args.target}_{'{t+1}' if args.target_shift else '{t}'}"
    )

    if args.raw_first_n > 0:
        raw_payload = collect_first_n_by_adapter(rows, p_cols, e_cols, args.raw_first_n)
        out_png = args.outdir / f"temporal_first{args.raw_first_n}_{suffix}.png"
        draw_first_n_figure(
            raw_payload=raw_payload,
            out_png=out_png,
            title=f"{title} (first {args.raw_first_n} raw trajectories)",
            first_n=args.raw_first_n,
        )
    else:
        draw_clean_figure(
            p_payload=p_payload,
            e_payload=e_payload,
            out_png=out_png,
            title=title,
            target=args.target,
        )
    write_summary(
        p_payload=p_payload,
        e_payload=e_payload,
        out_md=out_md,
        title=title,
    )

    print("[DONE]")
    print(" ", out_png)
    print(" ", out_png.with_suffix(".pdf"))
    print(" ", out_md)


if __name__ == "__main__":
    main()
