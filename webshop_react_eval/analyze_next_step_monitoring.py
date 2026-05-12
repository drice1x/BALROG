#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def safe_mean(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def load_rows(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def collect_pairs(rows):
    pairs = []
    for i in range(len(rows) - 1):
        cur = rows[i]
        nxt = rows[i + 1]
        if cur.get("episode_id") != nxt.get("episode_id"):
            continue
        pairs.append(
            {
                "model": cur.get("model"),
                "episode_id": cur.get("episode_id"),
                "task_id": cur.get("task_id"),
                "step_t": cur.get("step"),
                "p_hack_t": cur.get("p_hack"),
                "entropy_t": cur.get("entropy_mean"),
                "action_t1": nxt.get("validated_action"),
                "buy_t1": int(nxt.get("buy_action", 0) or 0),
                "bad_buy_t1": int(nxt.get("bad_buy_action", 0) or 0),
                "low_reward_buy_t1": int(nxt.get("low_reward_buy", 0) or 0),
                "early_buy_t1": int(nxt.get("early_buy", 0) or 0),
            }
        )
    return pairs


def summarize_model(model: str, pairs: list[dict]):
    def filt(key):
        return [row for row in pairs if row.get(key, 0) == 1]

    buy = filt("buy_t1")
    bad = filt("bad_buy_t1")
    low = filt("low_reward_buy_t1")
    non_buy = [row for row in pairs if row.get("buy_t1", 0) == 0]
    non_bad = [row for row in pairs if row.get("bad_buy_t1", 0) == 0]

    return {
        "model": model,
        "n_pairs": len(pairs),
        "n_buy_t1": len(buy),
        "n_bad_buy_t1": len(bad),
        "n_low_reward_buy_t1": len(low),
        "mean_p_hack_t": safe_mean([r["p_hack_t"] for r in pairs]),
        "mean_entropy_t": safe_mean([r["entropy_t"] for r in pairs]),
        "mean_p_hack_t__buy_t1": safe_mean([r["p_hack_t"] for r in buy]),
        "mean_entropy_t__buy_t1": safe_mean([r["entropy_t"] for r in buy]),
        "mean_p_hack_t__nonbuy_t1": safe_mean([r["p_hack_t"] for r in non_buy]),
        "mean_entropy_t__nonbuy_t1": safe_mean([r["entropy_t"] for r in non_buy]),
        "mean_p_hack_t__bad_buy_t1": safe_mean([r["p_hack_t"] for r in bad]),
        "mean_entropy_t__bad_buy_t1": safe_mean([r["entropy_t"] for r in bad]),
        "mean_p_hack_t__nonbad_buy_t1": safe_mean([r["p_hack_t"] for r in non_bad]),
        "mean_entropy_t__nonbad_buy_t1": safe_mean([r["entropy_t"] for r in non_bad]),
        "mean_p_hack_t__low_reward_buy_t1": safe_mean([r["p_hack_t"] for r in low]),
        "mean_entropy_t__low_reward_buy_t1": safe_mean([r["entropy_t"] for r in low]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    model_summaries = []
    for model_dir in sorted(runs_root.iterdir()):
        step_path = model_dir / "steps" / "steps.jsonl"
        if not step_path.exists():
            continue
        rows = load_rows(step_path)
        pairs = collect_pairs(rows)
        all_pairs.extend(pairs)
        model_summaries.append(summarize_model(model_dir.name, pairs))

    (outdir / "next_step_pairs.json").write_text(json.dumps(all_pairs, indent=2), encoding="utf-8")
    (outdir / "model_summary.json").write_text(json.dumps(model_summaries, indent=2), encoding="utf-8")

    headers = [
        "model",
        "n_pairs",
        "n_buy_t1",
        "n_bad_buy_t1",
        "mean_p_hack_t",
        "mean_entropy_t",
        "mean_p_hack_t__buy_t1",
        "mean_p_hack_t__bad_buy_t1",
        "mean_p_hack_t__low_reward_buy_t1",
        "mean_entropy_t__buy_t1",
        "mean_entropy_t__bad_buy_t1",
        "mean_entropy_t__low_reward_buy_t1",
    ]
    print(json.dumps(model_summaries, indent=2))
    print("\t".join(headers))
    for row in model_summaries:
        print("\t".join(str(row.get(h, "")) for h in headers))


if __name__ == "__main__":
    main()
