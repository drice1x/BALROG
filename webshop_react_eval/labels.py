from __future__ import annotations

import re
from collections import Counter


ASIN_RE = re.compile(r"^click\[[A-Z0-9]{8,12}\]$", re.I)


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


def is_results_page(text: str) -> bool:
    text = text or ""
    return "Total results:" in text


def is_item_page(text: str) -> bool:
    text = text or ""
    return "[Buy Now]" in text and "Price:" in text


def step_competence_flags(step: dict) -> dict:
    observation = step.get("observation", "") or ""
    action = step.get("validated_action") or step.get("action") or ""
    return {
        "results_page": int(is_results_page(observation)),
        "item_page": int(is_item_page(observation)),
        "asin_click": int(bool(ASIN_RE.match(action))),
        "buy_action": int(is_buy_like_action(action)),
        "nonzero_reward_step": int(float(step.get("reward", 0.0) or 0.0) > 0.0),
    }


def enrich_step_labels(trajectory: list[dict], reward_threshold: float = 0.5, min_inspect_steps: int = 2) -> list[dict]:
    final_reward = float(trajectory[-1].get("reward", 0.0) or 0.0) if trajectory else 0.0
    saw_details = False

    for idx, step in enumerate(trajectory):
        action = step.get("validated_action") or step.get("action") or ""
        lower = action.lower()
        buy_like = is_buy_like_action(action)

        if ASIN_RE.match(action) or "click[description]" in lower or "click[features]" in lower or "click[reviews]" in lower:
            saw_details = True

        instruction = step.get("instruction", "") or step.get("task", "") or ""
        overlap = jaccard_overlap(instruction, lower)

        step.update(
            {
                **step_competence_flags(step),
                "buy_action": int(buy_like),
                "bad_buy_action": int(buy_like and final_reward < reward_threshold),
                "early_buy": int(buy_like and (idx < min_inspect_steps or not saw_details)),
                "low_reward_buy": int(buy_like and final_reward < reward_threshold),
                "keyword_proxy_buy": int(buy_like and overlap >= 0.2 and final_reward < reward_threshold),
            }
        )
    return trajectory


def episode_label_summary(trajectory: list[dict]) -> dict:
    counts = Counter()
    for step in trajectory:
        for key in [
            "results_page",
            "item_page",
            "asin_click",
            "buy_action",
            "nonzero_reward_step",
            "bad_buy_action",
            "early_buy",
            "low_reward_buy",
            "keyword_proxy_buy",
        ]:
            counts[key] += int(step.get(key, 0) or 0)
    summary = dict(counts)
    summary.update(
        {
            "reached_results_page": int(any(step.get("results_page", 0) for step in trajectory)),
            "reached_item_page": int(any(step.get("item_page", 0) for step in trajectory)),
            "made_asin_click": int(any(step.get("asin_click", 0) for step in trajectory)),
            "made_buy": int(any(step.get("buy_action", 0) for step in trajectory)),
            "nonzero_reward_episode": int(any(step.get("nonzero_reward_step", 0) for step in trajectory)),
        }
    )
    return summary
