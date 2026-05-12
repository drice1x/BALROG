#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, lr: float = 0.1, steps: int = 2000) -> np.ndarray:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    for _ in range(steps):
        p = sigmoid(X @ w)
        grad = (X.T @ (p - y)) / max(n, 1) + l2 * w
        w -= lr * grad
    return w


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train - mean) / std, (test - mean) / std


def auc_roc(y: np.ndarray, score: np.ndarray) -> float | None:
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    rank_sum = ranks[pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    total_pos = int((y == 1).sum())
    if total_pos == 0:
        return None
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = 0
    fp = 0
    precisions = []
    for label in y_sorted:
        if label == 1:
            tp += 1
            precisions.append(tp / max(tp + fp, 1))
        else:
            fp += 1
    return float(sum(precisions) / total_pos) if precisions else 0.0


def recall_at_flag_rate(y: np.ndarray, score: np.ndarray, flag_rate: float) -> dict:
    n = len(y)
    k = max(1, int(np.ceil(flag_rate * n)))
    order = np.argsort(-score)
    flagged = order[:k]
    positives = (y == 1)
    n_pos = int(positives.sum())
    if n_pos == 0:
        return {"flag_rate": flag_rate, "k": k, "recall": None, "precision": None}
    tp = int(positives[flagged].sum())
    precision = tp / k
    recall = tp / n_pos
    return {"flag_rate": flag_rate, "k": k, "recall": recall, "precision": precision}


def load_pairs(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_dataset(rows: list[dict], target: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    X = []
    y = []
    groups = []
    models = []
    for row in rows:
        if row.get("p_hack_t") is None or row.get("entropy_t") is None:
            continue
        X.append([float(row["p_hack_t"]), float(row["entropy_t"])])
        y.append(int(row[target]))
        groups.append(f"{row['model']}::{row['episode_id']}")
        models.append(str(row["model"]))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), groups, models


def oof_predict(X: np.ndarray, y: np.ndarray, groups: list[str]) -> np.ndarray:
    preds = np.zeros(len(y), dtype=float)
    unique_groups = sorted(set(groups))
    group_arr = np.asarray(groups)
    for group in unique_groups:
        test_mask = group_arr == group
        train_mask = ~test_mask
        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]
        if len(np.unique(y_train)) < 2:
            preds[test_mask] = float(y_train[0]) if len(y_train) else 0.0
            continue
        X_train_s, X_test_s = standardize(X_train, X_test)
        X_train_aug = np.concatenate([np.ones((len(X_train_s), 1)), X_train_s], axis=1)
        X_test_aug = np.concatenate([np.ones((len(X_test_s), 1)), X_test_s], axis=1)
        w = fit_logreg(X_train_aug, y_train)
        preds[test_mask] = sigmoid(X_test_aug @ w)
    return preds


def summarize_split(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "positives": int((y == 1).sum()),
        "base_rate": float((y == 1).mean()) if len(y) else None,
        "auroc": auc_roc(y, score),
        "average_precision": average_precision(y, score),
        "recall_at_10pct_flag": recall_at_flag_rate(y, score, 0.10),
        "recall_at_20pct_flag": recall_at_flag_rate(y, score, 0.20),
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-json", default="runs/next_step_analysis_buycurated/next_step_pairs.json")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    rows = load_pairs(Path(args.pairs_json))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    targets = ["buy_t1", "bad_buy_t1", "low_reward_buy_t1"]
    results = {}

    for target in targets:
        X, y, groups, models = build_dataset(rows, target)
        score = oof_predict(X, y, groups)
        pooled = summarize_split(y, score)

        by_model = {}
        model_arr = np.asarray(models)
        for model in sorted(set(models)):
            mask = model_arr == model
            by_model[model] = summarize_split(y[mask], score[mask])

        results[target] = {
            "pooled": pooled,
            "by_model": by_model,
        }

    (outdir / "predictive_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "| Target | Split | N | Positives | Base Rate | AUROC | AP | Recall@10% flag | Recall@20% flag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, blob in results.items():
        pooled = blob["pooled"]
        lines.append(
            f"| {target} | pooled | {pooled['n']} | {pooled['positives']} | {fmt(pooled['base_rate'])} | "
            f"{fmt(pooled['auroc'])} | "
            f"{fmt(pooled['average_precision'])} | "
            f"{fmt(pooled['recall_at_10pct_flag']['recall'])} | "
            f"{fmt(pooled['recall_at_20pct_flag']['recall'])} |"
        )
        for model, summary in blob["by_model"].items():
            lines.append(
                f"| {target} | {model} | {summary['n']} | {summary['positives']} | {fmt(summary['base_rate'])} | "
                f"{fmt(summary['auroc'])} | "
                f"{fmt(summary['average_precision'])} | "
                f"{fmt(summary['recall_at_10pct_flag']['recall'])} | "
                f"{fmt(summary['recall_at_20pct_flag']['recall'])} |"
            )

    markdown = "\n".join(lines) + "\n"
    (outdir / "predictive_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
