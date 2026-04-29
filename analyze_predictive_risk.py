#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = [
    "gameable_hack_action",
    "fake_completion_action",
    "easy_grader_action",
    "proxy_farming_action",
]

FEATURE_GROUPS = {
    "mean_phack": [
        "reasoning_p_hack",
        "action_p_hack",
    ],
    "late_phack": [
        "reasoning_p_hack_late_slope",
        "reasoning_p_hack_late_change",
        "action_p_hack_late_slope",
        "action_p_hack_late_change",
    ],
    "entropy": [
        "reasoning_entropy_mean",
        "action_entropy_mean",
        "reasoning_entropy_late_slope",
        "reasoning_entropy_late_change",
        "action_entropy_late_slope",
        "action_entropy_late_change",
    ],
    "all": [
        "reasoning_p_hack",
        "action_p_hack",
        "reasoning_p_hack_late_slope",
        "reasoning_p_hack_late_change",
        "action_p_hack_late_slope",
        "action_p_hack_late_change",
        "reasoning_entropy_mean",
        "action_entropy_mean",
        "reasoning_entropy_late_slope",
        "reasoning_entropy_late_change",
        "action_entropy_late_slope",
        "action_entropy_late_change",
    ],
}


def sem(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) <= 1:
        return 0.0
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def make_future_target(df: pd.DataFrame, target: str, horizon: int) -> pd.Series:
    out = pd.Series(0, index=df.index, dtype=int)

    for _, sub in df.groupby("source_file", dropna=False):
        idx = sub.sort_values("step").index.to_list()
        vals = sub.loc[idx, target].fillna(0).astype(int).to_list()

        future = []
        for i in range(len(vals)):
            j0 = i + 1
            j1 = min(len(vals), i + 1 + horizon)
            future.append(int(any(vals[j0:j1])))

        out.loc[idx] = future

    return out


def evaluate_leave_one_episode_out(df: pd.DataFrame, target: str, features: list[str]):
    needed = ["source_file", target] + features
    d = df[needed].replace([np.inf, -np.inf], np.nan).dropna()

    if d.empty or d[target].nunique() < 2:
        return None

    y_all = []
    score_all = []

    for holdout in sorted(d["source_file"].unique()):
        train = d[d["source_file"] != holdout]
        test = d[d["source_file"] == holdout]

        if train[target].nunique() < 2 or test.empty:
            continue

        X_train = train[features].values
        y_train = train[target].astype(int).values
        X_test = test[features].values
        y_test = test[target].astype(int).values

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
            ),
        )

        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)[:, 1]

        y_all.extend(y_test.tolist())
        score_all.extend(scores.tolist())

    if len(set(y_all)) < 2:
        return None

    return {
        "n": len(y_all),
        "positives": int(sum(y_all)),
        "base_rate": float(np.mean(y_all)),
        "auroc": float(roc_auc_score(y_all, score_all)),
        "auprc": float(average_precision_score(y_all, score_all)),
    }


def signal_deltas(df: pd.DataFrame, target: str, features: list[str]):
    rows = []

    for feat in features:
        if feat not in df.columns:
            continue

        good = pd.to_numeric(df[df[target] == 0][feat], errors="coerce").dropna()
        bad = pd.to_numeric(df[df[target] == 1][feat], errors="coerce").dropna()

        if len(good) == 0 or len(bad) == 0:
            continue

        pooled = pd.concat([good, bad])
        pooled_std = pooled.std(ddof=1)
        effect = None
        if pooled_std and not np.isnan(pooled_std) and pooled_std > 0:
            effect = float((bad.mean() - good.mean()) / pooled_std)

        rows.append(
            {
                "feature": feat,
                "n_good": len(good),
                "n_bad": len(bad),
                "good_mean": float(good.mean()),
                "bad_mean": float(bad.mean()),
                "delta_bad_minus_good": float(bad.mean() - good.mean()),
                "standardized_effect": effect,
            }
        )

    return rows


def plot_cv_results(cv: pd.DataFrame, outpath: Path):
    if cv.empty:
        return

    d = cv.copy()
    d = d[d["target"].str.startswith("future_")]
    if d.empty:
        d = cv.copy()

    g = (
        d.groupby(["target", "feature_group"], as_index=False)
        .agg(
            auroc=("auroc", "mean"),
            auprc=("auprc", "mean"),
            n=("n", "sum"),
            positives=("positives", "sum"),
        )
    )

    for metric in ["auroc", "auprc"]:
        plt.figure(figsize=(11, 5))
        labels = []
        values = []

        for _, r in g.iterrows():
            labels.append(f"{r['target']}\n{r['feature_group']}")
            values.append(r[metric])

        plt.bar(range(len(values)), values)
        plt.axhline(0.5 if metric == "auroc" else 0.0, linestyle="--", linewidth=1)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"Predictive risk model: {metric.upper()}")
        plt.tight_layout()

        p = outpath / f"predictive_risk_{metric}.png"
        plt.savefig(p, dpi=200)
        plt.close()
        print(f"[OK] saved {p}")


def plot_feature_deltas(delta: pd.DataFrame, outpath: Path):
    if delta.empty:
        return

    d = delta.copy()
    d = d[d["target"].str.startswith("future_")]
    if d.empty:
        d = delta.copy()

    g = (
        d.groupby("feature", as_index=False)
        .agg(effect=("standardized_effect", "mean"))
        .sort_values("effect", ascending=False)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(g["feature"], g["effect"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Standardized effect: event minus non-event")
    plt.title("Predictive signal strength")
    plt.tight_layout()

    p = outpath / "predictive_signal_effects.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"[OK] saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--steps-csv",
        type=Path,
        default=Path("analysis_gameable_alfworld_temporal/steps_with_entropy_phack_temporal.csv"),
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("analysis_predictive_risk"),
    )
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--include-react-baseline", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.steps_csv)

    if not args.include_react_baseline and "method" in df.columns:
        df = df[df["method"] == "react_ttc_monitored"].copy()

    for target in TARGETS:
        if target in df.columns:
            df[f"future_{target}"] = make_future_target(df, target, args.horizon)

    group_cols = ["model", "adapter", "task", "ttc"]

    cv_rows = []
    delta_rows = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))

        for raw_target in TARGETS:
            if raw_target not in sub.columns:
                continue

            targets_to_eval = [raw_target, f"future_{raw_target}"]

            for target in targets_to_eval:
                if target not in sub.columns:
                    continue
                if sub[target].fillna(0).astype(int).nunique() < 2:
                    continue

                for group_name, feats in FEATURE_GROUPS.items():
                    feats = [f for f in feats if f in sub.columns]
                    if not feats:
                        continue

                    res = evaluate_leave_one_episode_out(sub, target, feats)
                    if res is not None:
                        cv_rows.append(
                            {
                                **base,
                                "target": target,
                                "feature_group": group_name,
                                **res,
                            }
                        )

                    for r in signal_deltas(sub, target, feats):
                        delta_rows.append(
                            {
                                **base,
                                "target": target,
                                "feature_group": group_name,
                                **r,
                            }
                        )

    cv = pd.DataFrame(cv_rows)
    delta = pd.DataFrame(delta_rows)

    cv.to_csv(args.outdir / "predictive_risk_cv.csv", index=False)
    delta.to_csv(args.outdir / "predictive_signal_deltas.csv", index=False)

    plot_cv_results(cv, args.outdir)
    plot_feature_deltas(delta, args.outdir)

    print("\n=== Predictive CV ===")
    print(cv.to_string(index=False) if not cv.empty else "No valid CV rows.")

    print("\n=== Signal Deltas ===")
    print(delta.head(100).to_string(index=False) if not delta.empty else "No valid delta rows.")

    print(f"\nSaved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()