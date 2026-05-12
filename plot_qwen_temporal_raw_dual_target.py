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


def target_value(row: dict, target: str) -> float:
    if target == "bad_action":
        return fnum(row.get("bad_action")) or 0.0
    if target == "exploit_action":
        vals = [
            fnum(row.get("gameable_hack_action")) or 0.0,
            fnum(row.get("easy_grader_action")) or 0.0,
            fnum(row.get("fake_completion_action")) or 0.0,
        ]
        return 1.0 if sum(vals) > 0 else 0.0
    return fnum(row.get(target)) or 0.0


def load_base_rows(
    steps_csv: Path,
    family: str,
    method: str,
    ttc_filter: Optional[str],
) -> tuple[list[str], list[str], list[dict]]:
    rows = read_rows(steps_csv)
    if not rows:
        raise RuntimeError("Empty CSV.")
    p_cols = bin_cols(rows, "reasoning_p_bin")
    e_cols = bin_cols(rows, "reasoning_entropy_bin")
    if not p_cols or not e_cols:
        raise RuntimeError("Missing reasoning_p_bin* or reasoning_entropy_bin* columns.")

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
    return p_cols, e_cols, out


def attach_shifted_labels(rows: list[dict], target: str, target_shift: int) -> list[dict]:
    out = [dict(r) for r in rows]
    for r in out:
        r["_target_raw"] = target_value(r, target)
    for i, r in enumerate(out):
        label = None
        if target_shift == 0:
            label = r["_target_raw"]
        else:
            j = i + target_shift
            if j < len(out) and out[j].get("source_file") == r.get("source_file"):
                label = out[j]["_target_raw"]
        r["_risk_label"] = label
    return out


def collect_first_n_by_adapter(rows: list[dict], p_cols: list[str], e_cols: list[str], first_n: int) -> dict:
    payload = {}
    valid = []
    for r in rows:
        if r["_risk_label"] is None:
            continue
        p_vals = [fnum(r.get(c)) for c in p_cols]
        e_vals = [fnum(r.get(c)) for c in e_cols]
        if any(v is None for v in p_vals) or any(v is None for v in e_vals):
            continue
        valid.append(r)
    for adapter in ADAPTERS:
        sub = [r for r in valid if r.get("adapter") == adapter]
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
        payload[adapter] = {
            "traces": selected,
            "n_valid": len(sub),
            "n_bad": sum(int(float(r["_risk_label"]) > 0.5) for r in sub),
            "n_nonbad": sum(int(float(r["_risk_label"]) <= 0.5) for r in sub if r["_risk_label"] is not None),
        }
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


def draw_dual_target_raw_figure(
    bad_payload: dict,
    exploit_payload: dict,
    out_png: Path,
    title: str,
    first_n: int,
):
    set_style()
    x = np.linspace(0.0, 1.0, 8)
    fig, axes = plt.subplots(
        nrows=len(ADAPTERS),
        ncols=4,
        figsize=(14.0, 9.8),
        sharex=True,
        constrained_layout=True,
    )

    col_titles = [
        r"Bad next action: $p_{\mathrm{hack}}$",
        "Bad next action: entropy",
        r"Exploit next action: $p_{\mathrm{hack}}$",
        "Exploit next action: entropy",
    ]
    for col, ctitle in enumerate(col_titles):
        axes[0, col].set_title(ctitle, fontweight="bold")

    for r_idx, adapter in enumerate(ADAPTERS):
        cell_specs = [
            (bad_payload[adapter], "p_vals"),
            (bad_payload[adapter], "e_vals"),
            (exploit_payload[adapter], "p_vals"),
            (exploit_payload[adapter], "e_vals"),
        ]
        for c_idx, (cell, key) in enumerate(cell_specs):
            ax = axes[r_idx, c_idx]
            ax.axvspan(0.8, 1.0, color="#dddddd", alpha=0.35, zorder=0)
            ax.grid(axis="y")
            traces = cell["traces"]
            if not traces:
                ax.text(
                    0.5,
                    0.5,
                    f"no trajectories\nvalid={cell['n_valid']}\nbad={cell['n_bad']}\nnon={cell['n_nonbad']}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            for i, tr in enumerate(traces):
                vals = tr[key]
                color = BAD_COLOR if tr["risk_label"] == 1 else GOOD_COLOR
                ls = "-" if tr["risk_label"] == 1 else "--"
                alpha = max(0.35, 0.9 - i * 0.1)
                ax.plot(x, vals, color=color, linestyle=ls, linewidth=1.5, alpha=alpha)
                ax.scatter([x[-1]], [vals[-1]], color=color, s=10, alpha=alpha)
            if c_idx == 0:
                ax.set_ylabel(
                    f"{ADAPTER_LABELS[adapter]}\n"
                    f"first {min(first_n, len(traces))} traces\n"
                    f"bad: valid={bad_payload[adapter]['n_valid']}\n"
                    f"exploit: valid={exploit_payload[adapter]['n_valid']}"
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


def write_note(out_md: Path, family: str, method: str, ttc: Optional[str], first_n: int) -> None:
    lines = [
        f"# {family} raw temporal traces: bad_action vs exploit_action",
        "",
        f"- family: `{family}`",
        f"- method filter: `{method}`",
        f"- ttc filter: `{ttc or 'all available in filter'}`",
        f"- first raw trajectories per adapter: `{first_n}`",
        "- columns 1-2 condition on `bad_action_{t+1}`",
        "- columns 3-4 condition on `exploit_action_{t+1}`",
        "- odd metric columns show `p_hack`; even metric columns show entropy",
        "",
        "This figure is raw, not aggregated. It is for direct inspection of reasoning-trace behavior patterns.",
    ]
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
    parser.add_argument("--target-shift", type=int, default=1)
    parser.add_argument("--ttc", type=str, default=None, help="Optional e.g. ttc_32, ttc_64")
    parser.add_argument("--outdir", type=Path, default=Path("paper_figures"))
    parser.add_argument("--raw-first-n", type=int, default=5)
    args = parser.parse_args()

    p_cols, e_cols, base_rows = load_base_rows(
        steps_csv=args.steps_csv,
        family=args.family,
        method=args.method,
        ttc_filter=args.ttc,
    )

    if not base_rows:
        raise RuntimeError("No rows after filtering.")

    bad_rows = attach_shifted_labels(base_rows, "bad_action", args.target_shift)
    exploit_rows = attach_shifted_labels(base_rows, "exploit_action", args.target_shift)
    bad_payload = collect_first_n_by_adapter(bad_rows, p_cols, e_cols, args.raw_first_n)
    exploit_payload = collect_first_n_by_adapter(exploit_rows, p_cols, e_cols, args.raw_first_n)

    suffix = f"{args.family.lower()}_bad_vs_exploit_shift{args.target_shift}"
    if args.ttc:
        suffix += f"_{args.ttc}"
    out_png = args.outdir / f"temporal_dual_target_first{args.raw_first_n}_{suffix}.png"
    out_md = args.outdir / f"temporal_dual_target_first{args.raw_first_n}_{suffix}.md"

    draw_dual_target_raw_figure(
        bad_payload=bad_payload,
        exploit_payload=exploit_payload,
        out_png=out_png,
        title=f"{args.family}: first {args.raw_first_n} raw reasoning traces before bad vs exploit next actions",
        first_n=args.raw_first_n,
    )
    write_note(out_md, args.family, args.method, args.ttc, args.raw_first_n)

    print("[DONE]")
    print(" ", out_png)
    print(" ", out_png.with_suffix(".pdf"))
    print(" ", out_md)


if __name__ == "__main__":
    main()
