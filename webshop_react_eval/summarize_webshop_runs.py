#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    rows = []
    for path in sorted(runs_root.glob("*/pilot_metrics.json")):
        metrics = json.loads(path.read_text())
        metrics["run_dir"] = str(path.parent)
        rows.append(metrics)

    print(json.dumps(rows, indent=2))

    if not rows:
        return

    headers = [
        "model",
        "episodes",
        "avg_episode_return",
        "results_page_reach_rate",
        "item_page_reach_rate",
        "asin_click_rate",
        "buy_rate",
        "nonzero_reward_rate",
        "bad_buy_rate",
        "low_reward_buy_rate",
    ]
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row.get(h, "")) for h in headers))


if __name__ == "__main__":
    main()
