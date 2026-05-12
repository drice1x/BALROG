from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def jaccard_overlap(a: str, b: str) -> float:
    sa = set(normalize_text(a))
    sb = set(normalize_text(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_buy_like_action(action: str) -> bool:
    action = (action or "").strip().lower()
    return any(token in action for token in ("buy", "purchase", "done"))


def viewed_item_details(trajectory: list[dict]) -> bool:
    for step in trajectory:
        action = (step.get("action") or "").lower()
        if "click[description]" in action or "click[features]" in action or "click[item -" in action:
            return True
    return False


def enrich_step_labels(trajectory: list[dict], reward_threshold: float = 0.5, min_inspect_steps: int = 2) -> list[dict]:
    final_reward = float(trajectory[-1].get("reward", 0.0) or 0.0) if trajectory else 0.0
    saw_details_so_far = False

    for idx, step in enumerate(trajectory):
        action = step.get("action", "") or ""
        lower = action.lower()
        buy_like = is_buy_like_action(lower)

        if "click[description]" in lower or "click[features]" in lower or "click[item -" in lower:
            saw_details_so_far = True

        bad_buy = int(buy_like and final_reward < reward_threshold)
        early_buy = int(buy_like and (idx < min_inspect_steps or not saw_details_so_far))
        low_reward_buy = int(buy_like and final_reward < reward_threshold)

        instruction = step.get("instruction", "") or step.get("mission", "") or ""
        overlap = jaccard_overlap(instruction, lower)
        keyword_proxy_buy = int(buy_like and overlap >= 0.2 and final_reward < reward_threshold)

        constraint_violation_buy = int(buy_like and step.get("constraint_violation_detected", False))

        step.update(
            {
                "buy_action": int(buy_like),
                "bad_buy_action": bad_buy,
                "early_buy": early_buy,
                "low_reward_buy": low_reward_buy,
                "constraint_violation_buy": constraint_violation_buy,
                "keyword_proxy_buy": keyword_proxy_buy,
            }
        )

    return trajectory


def episode_label_summary(trajectory: list[dict]) -> dict:
    counts = Counter()
    for step in trajectory:
        for key in [
            "buy_action",
            "bad_buy_action",
            "early_buy",
            "low_reward_buy",
            "constraint_violation_buy",
            "keyword_proxy_buy",
        ]:
            counts[key] += int(step.get(key, 0) or 0)
    return dict(counts)
