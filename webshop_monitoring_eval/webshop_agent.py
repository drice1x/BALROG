from __future__ import annotations

from dataclasses import dataclass
import re
from difflib import get_close_matches

from monitoring_client import MonitoringClient, MonitoringResponse


@dataclass
class AgentStep:
    action: str
    raw_action: str
    reasoning_text: str
    action_text: str
    reasoning_trace: MonitoringResponse
    action_trace: MonitoringResponse
    search_query: str | None
    zero_results_page: bool
    search_retry_count: int
    used_search_retry: bool
    action_projection_method: str
    invalid_action: bool


def safe_float(value):
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def late_change(values: list[float], frac: float = 0.1):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    k = max(1, int(round(len(vals) * frac)))
    return (sum(vals[-k:]) / k) - (sum(vals[:k]) / k)


def late_slope(values: list[float], frac: float = 0.2):
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if len(vals) < 3:
        return None
    k = max(2, int(round(len(vals) * frac)))
    ys = vals[-k:]
    xs = list(range(len(ys)))
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom


def build_search_query(instruction: str) -> str:
    """Generic fallback only for empty/placeholder search actions."""
    text = (instruction or "").replace("[SEP]", " ")
    text = re.sub(r"(?i)\b(webshop|instruction|search)\b", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 /-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Keep this generic; do not use product-category rules.
    return " ".join(text.split()[:10]) or "product"


def broaden_search_query(instruction: str, prior_query: str, retry_count: int) -> str:
    instruction_query = build_search_query(instruction)
    instruction_tokens = instruction_query.split()
    prior_tokens = [tok for tok in (prior_query or "").split() if tok]
    tokens = prior_tokens or instruction_tokens
    if not tokens:
        return instruction_query

    if retry_count <= 1:
        keep = max(3, len(tokens) - 2)
        broader = tokens[:keep]
    elif retry_count == 2:
        keep = max(3, len(tokens) // 2)
        broader = tokens[:keep]
    else:
        broader = instruction_tokens[: max(3, min(5, len(instruction_tokens)))] or tokens[:3]

    query = " ".join(broader).strip() or instruction_query
    if query.lower() == (prior_query or "").strip().lower() and len(tokens) > 3:
        query = " ".join(tokens[: len(tokens) - 1]).strip()
    return query or instruction_query


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def query_similarity(a: str, b: str) -> float:
    a_tokens = set(normalized_tokens(a))
    b_tokens = set(normalized_tokens(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def next_broader_query(instruction: str, failed_queries: list[str], retry_index: int) -> str:
    instruction_query = build_search_query(instruction)
    instruction_tokens = normalized_tokens(instruction_query)
    last_query = failed_queries[-1] if failed_queries else instruction_query
    last_tokens = normalized_tokens(last_query) or instruction_tokens

    candidate_token_lists: list[list[str]] = []
    if last_tokens:
        candidate_token_lists.append(last_tokens[: max(3, len(last_tokens) - 2)])
        candidate_token_lists.append(last_tokens[: max(3, len(last_tokens) // 2)])
        candidate_token_lists.append(last_tokens[:3])
    if instruction_tokens:
        candidate_token_lists.append(instruction_tokens[: max(3, min(6, len(instruction_tokens)))])
        candidate_token_lists.append(instruction_tokens[: max(2, min(4, len(instruction_tokens)))])

    seen = [" ".join(normalized_tokens(q)) for q in failed_queries if normalized_tokens(q)]
    offset = max(0, retry_index - 1)
    for idx in range(len(candidate_token_lists)):
        tokens = candidate_token_lists[min(idx + offset, len(candidate_token_lists) - 1)]
        candidate = " ".join(tokens).strip()
        if not candidate:
            continue
        if all(query_similarity(candidate, prior) < 0.8 for prior in seen):
            return candidate

    fallback = " ".join(instruction_tokens[: max(2, min(3, len(instruction_tokens)))]) if instruction_tokens else ""
    if fallback and all(query_similarity(fallback, prior) < 0.95 for prior in seen):
        return fallback
    return instruction_query


ASIN_RE = re.compile(r"^[A-Z0-9]{8,12}$", flags=re.IGNORECASE)


REACT_WEBSHOP_FEWSHOT = """WebShop
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

Thought: I should search with the core product attributes, then inspect a promising item and choose the matching options.
Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B078GWRC1J]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Contains Organic Calendula 3-Ounce
$10.99
[B078GTKVXY]
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Contains Organic Calendula 3-Ounce
$10.99

Thought: B078GWRC1J matches bright citrus, sensitive skin, and the price constraint. I should inspect it.
Action: click[B078GWRC1J]
Observation:
[Back to Search]
[< Prev]
scent [assorted scents][bright citrus][calming lavender][ginger fresh]
size [travel set (4-pack)][3 ounce (pack of 1)][3-ounce (2-pack)]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Contains Organic Calendula 3-Ounce
Price: $10.99
[Description]
[Features]
[Reviews]
[Buy Now]

Thought: The item page is relevant. I should click the exact matching options before buying.
Action: click[bright citrus]
Observation: You have clicked bright citrus.

Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).

Thought: The constraints are satisfied, so buying is appropriate.
Action: click[Buy Now]
"""


class WebShopReactAgent:
    def __init__(
        self,
        client: MonitoringClient,
        reasoning_max_tokens: int = 32,
        action_max_tokens: int = 16,
        reasoning_temperature: float = 0.7,
        action_temperature: float = 0.0,
        ttc_mode: bool = True,
        agent_mode: str = "monitored_ttc",
    ):
        self.client = client
        self.reasoning_max_tokens = reasoning_max_tokens
        self.action_max_tokens = action_max_tokens
        self.reasoning_temperature = reasoning_temperature
        self.action_temperature = action_temperature
        self.ttc_mode = ttc_mode
        self.agent_mode = agent_mode
        self.history: list[dict[str, str]] = []
        self.search_retry_count = 0
        self.pending_search_query: str | None = None
        self.zero_result_streak = 0
        self.failed_search_queries: list[str] = []

    def reset(self):
        self.history = []
        self.search_retry_count = 0
        self.pending_search_query = None
        self.zero_result_streak = 0
        self.failed_search_queries = []

    def act(self, observation: dict) -> AgentStep:
        if self.agent_mode == "react_original":
            return self._act_react_original(observation)

        available_actions = observation.get("available_actions", []) or []
        zero_results_page = self._is_zero_results_page(observation)
        retry_loop = self._is_back_search_loop()
        used_search_retry = bool(zero_results_page or retry_loop)
        previous_search_query = self._last_search_query()

        if zero_results_page:
            self.zero_result_streak += 1
            if previous_search_query and (
                not self.failed_search_queries
                or self.failed_search_queries[-1].strip().lower() != previous_search_query.strip().lower()
            ):
                self.failed_search_queries.append(previous_search_query)
        elif self._is_results_page(observation, available_actions):
            self.zero_result_streak = 0
            self.failed_search_queries = []

        if used_search_retry:
            self.search_retry_count += 1
        else:
            self.search_retry_count = 0

        reasoning_prompt = self._build_reasoning_prompt(
            observation, available_actions, zero_results_page, used_search_retry, previous_search_query
        )
        reasoning_trace = self.client.complete(
            messages=[{"role": "user", "content": reasoning_prompt}],
            max_tokens=self.reasoning_max_tokens,
            temperature=self.reasoning_temperature,
            do_sample=self.reasoning_temperature > 0,
        )
        reasoning_text = reasoning_trace.text.strip()

        action_prompt = self._build_action_prompt(
            observation, available_actions, reasoning_text, zero_results_page, used_search_retry, previous_search_query
        )
        action_trace = self.client.complete(
            messages=[{"role": "user", "content": action_prompt}],
            max_tokens=self.action_max_tokens,
            temperature=self.action_temperature,
            do_sample=self.action_temperature > 0,
        )
        action_text = action_trace.text.strip()

        action, method, invalid = self._validate_action(action_text, available_actions, observation)
        action, method = self._apply_search_retry_policy(
            action,
            method,
            observation,
            available_actions,
            used_search_retry=used_search_retry,
        )
        action, method = self._apply_navigation_policy(
            action,
            method,
            observation,
            available_actions,
        )
        search_query = self._extract_search_query(action)

        self.history.append(
            {
                "observation": observation["text"]["long_term_context"],
                "reasoning": reasoning_text,
                "raw_action": action_text,
                "action": action,
            }
        )

        return AgentStep(
            action=action,
            raw_action=action_text,
            reasoning_text=reasoning_text,
            action_text=action_text,
            reasoning_trace=reasoning_trace,
            action_trace=action_trace,
            search_query=search_query,
            zero_results_page=zero_results_page,
            search_retry_count=self.search_retry_count,
            used_search_retry=used_search_retry,
            action_projection_method=method,
            invalid_action=invalid,
        )

    def _act_react_original(self, observation: dict) -> AgentStep:
        available_actions = observation.get("available_actions", []) or []
        prompt = self._build_original_react_prompt(observation, available_actions)
        trace = self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max(8, self.reasoning_max_tokens + self.action_max_tokens),
            temperature=self.reasoning_temperature,
            do_sample=self.reasoning_temperature > 0,
        )
        raw_text = trace.text.strip()
        reasoning_text, action_text = self._extract_thought_and_action(raw_text)
        action, method, invalid = self._validate_action(action_text, available_actions, observation)
        search_query = self._extract_search_query(action)

        self.history.append(
            {
                "observation": observation["text"]["long_term_context"],
                "reasoning": reasoning_text,
                "raw_action": raw_text,
                "action": action,
            }
        )

        return AgentStep(
            action=action,
            raw_action=raw_text,
            reasoning_text=reasoning_text,
            action_text=action_text,
            reasoning_trace=trace,
            action_trace=trace,
            search_query=search_query,
            zero_results_page=self._is_zero_results_page(observation),
            search_retry_count=0,
            used_search_retry=False,
            action_projection_method=method,
            invalid_action=invalid,
        )

    def _build_reasoning_prompt(
        self,
        observation,
        available_actions,
        zero_results_page,
        used_search_retry,
        previous_search_query,
    ) -> str:
        click_actions = [a for a in available_actions if a.lower().startswith("click[")]
        valid_clicks = "\n".join(f"- {a}" for a in click_actions[:80]) if click_actions else "- none"

        notes = []
        if self._is_search_page(observation, available_actions):
            notes.append("You are on the search page. Plan a concise search query.")
        if zero_results_page:
            notes.append("The previous search returned zero results. Broaden the query.")
        if previous_search_query:
            notes.append(f"Previous search query: {previous_search_query}")
        if self._is_results_page(observation, available_actions):
            notes.append("Search results are visible. Prefer clicking a promising product result instead of searching again.")
        if used_search_retry and not zero_results_page:
            notes.append("Recent search/back actions are looping. Do not repeat the same query.")

        note_block = "\n".join(f"- {n}" for n in notes) if notes else "- none"
        click_actions = [a for a in available_actions if a.lower().startswith("click[")]
        valid_clicks = "\n".join(f"- {a}" for a in click_actions[:80]) if click_actions else "- none"
        trajectory = self._format_recent_history()

        return (
            "You are a WebShop agent. Follow the ReAct pattern from the example.\n\n"
            f"{REACT_WEBSHOP_FEWSHOT}\n"
            "Now solve the next task.\n\n"
            "WebShop\n"
            f"Instruction:\n{observation.get('mission', '')}\n"
            f"{trajectory}"
            f"Observation:\n{observation['text']['long_term_context']}\n\n"
            "Valid action formats:\n"
            "- search[query]\n"
            "- click[exact option]\n\n"
            f"Clickable options:\n{valid_clicks}\n\n"
            f"Notes:\n{note_block}\n\n"
            "Write one short Thought about the next useful step.\n"
            "Thought:"
        )

    def _build_original_react_prompt(self, observation: dict, available_actions: list[str]) -> str:
        trajectory = self._format_recent_history()
        return (
            "Follow the WebShop examples and continue the trajectory.\n\n"
            f"{REACT_WEBSHOP_FEWSHOT}\n"
            "Now solve the next task.\n\n"
            "WebShop\n"
            f"Instruction:\n{observation.get('mission', '')}\n"
            f"{trajectory}"
            f"Observation:\n{observation['text']['long_term_context']}\n\n"
            "Respond with a short Thought line and then exactly one Action line.\n"
            "Valid actions are search[query] and click[exact option].\n"
            "If search results are visible, prefer clicking a promising product result.\n"
            "If on an item page, choose relevant options before Buy Now.\n"
            "Thought:"
        )

    def _build_action_prompt(
        self,
        observation,
        available_actions,
        reasoning_text,
        zero_results_page,
        used_search_retry,
        previous_search_query,
    ) -> str:
        click_actions = [a for a in available_actions if a.lower().startswith("click[")]
        valid_clicks = "\n".join(f"- {a}" for a in click_actions[:80]) if click_actions else "- none"
        trajectory = self._format_recent_history()

        notes = []
        if self._is_search_page(observation, available_actions):
            notes.append("If this is a search page, output search[query].")
        if zero_results_page:
            notes.append("Previous search returned zero results; use a broader search query.")
        if previous_search_query:
            notes.append(f"Do not repeat this previous search verbatim: {previous_search_query}")
        if self._is_results_page(observation, available_actions):
            notes.append("If product results are available, prefer clicking a product result such as an ASIN link over another search.")
        if used_search_retry and not zero_results_page:
            notes.append("Avoid repeating a search/back loop.")

        note_block = "\n".join(f"- {n}" for n in notes) if notes else "- none"

        return (
            "You are a WebShop agent. Continue the same ReAct trajectory format.\n\n"
            f"{REACT_WEBSHOP_FEWSHOT}\n"
            "Now continue the current episode.\n\n"
            "WebShop\n"
            f"Instruction:\n{observation.get('mission', '')}\n"
            f"{trajectory}"
            f"Observation:\n{observation['text']['long_term_context']}\n"
            f"Thought: {reasoning_text}\n\n"
            "Allowed formats:\n"
            "- search[query]\n"
            "- click[exact option]\n\n"
            f"Clickable options:\n{valid_clicks}\n\n"
            f"Notes:\n{note_block}\n\n"
            "Output exactly one Action line in the format Action: search[...] or Action: click[...].\n"
            "If item results are visible, prefer clicking a promising item.\n"
            "If you choose click, it must exactly match one listed clickable option.\n"
            "Action:"
        )

    def _validate_action(self, text: str, available_actions: list[str], observation: dict) -> tuple[str, str, bool]:
        raw_candidate = self._extract_action_candidate(text)
        instruction = observation.get("mission", "") or observation["text"]["long_term_context"]
        fallback_query = build_search_query(instruction)
        click_actions = [a for a in available_actions if a.lower().startswith("click[")]

        # 1. Preserve valid search actions. No projection to click[search].
        if self._is_search_action(raw_candidate):
            return self._normalize_search_action(raw_candidate, fallback_query), "search_direct", False

        # 2. Rewrite click[search] to real search query.
        if raw_candidate.lower() == "click[search]":
            return f"search[{fallback_query}]", "rewrite_click_search_to_search", False

        # 3. Exact/casefold click only.
        if raw_candidate in click_actions:
            return raw_candidate, "exact", False

        lower_actions = [a.lower() for a in click_actions]
        if raw_candidate.lower() in lower_actions:
            return click_actions[lower_actions.index(raw_candidate.lower())], "casefold_exact", False

        # 4. Only snap malformed click[...] to nearby valid click.
        if raw_candidate.lower().startswith("click[") and raw_candidate.endswith("]") and click_actions:
            target = raw_candidate[6:-1].strip().lower()
            click_targets = [a[6:-1].strip().lower() for a in click_actions]

            if target in click_targets:
                return click_actions[click_targets.index(target)], "click_target_exact", False

            match = get_close_matches(target, click_targets, n=1, cutoff=0.75)
            if match:
                return click_actions[click_targets.index(match[0])], "click_nearest", False

        # 5. Invalid generation. Do NOT silently choose first clickable.
        # For WebShop we need an executable action, so use a controlled search fallback and mark invalid.
        if self._is_search_page(observation, available_actions) or not click_actions:
            return f"search[{fallback_query}]", "invalid_fallback_search", True

        # If click actions exist but model emitted garbage, mark invalid and go back to search instead of selecting arbitrary click.
        return f"search[{fallback_query}]", "invalid_fallback_search_instead_of_click", True

    def _apply_search_retry_policy(
        self,
        action: str,
        method: str,
        observation: dict,
        available_actions: list[str],
        used_search_retry: bool,
    ) -> tuple[str, str]:
        if self.pending_search_query and self._is_search_page(observation, available_actions):
            current_query = self._extract_search_query(action) or ""
            pending_query = self.pending_search_query
            self.pending_search_query = None
            if not current_query or query_similarity(current_query, pending_query) >= 0.8:
                return f"search[{pending_query}]", "pending_broadened_search"

        if not used_search_retry or not self._is_search_action(action):
            return action, method

        current_query = self._extract_search_query(action) or ""
        previous_query = self._last_search_query()
        if not previous_query:
            return action, method

        same_query = current_query.strip().lower() == previous_query.strip().lower()
        near_duplicate = query_similarity(current_query, previous_query) >= 0.8
        if not same_query and not near_duplicate:
            return action, method

        instruction = observation.get("mission", "") or observation["text"]["long_term_context"]
        broader_query = next_broader_query(
            instruction,
            self.failed_search_queries + ([current_query] if current_query else []),
            max(self.zero_result_streak, self.search_retry_count, 1),
        )
        if broader_query.strip().lower() == current_query.strip().lower():
            return action, method
        if self._is_zero_results_page(observation) and self._has_back_to_search(available_actions):
            self.pending_search_query = broader_query
            return "click[back to search]", "zero_results_back_to_search"
        return f"search[{broader_query}]", "search_retry_broadened"

    def _apply_navigation_policy(
        self,
        action: str,
        method: str,
        observation: dict,
        available_actions: list[str],
    ) -> tuple[str, str]:
        if self._is_zero_results_page(observation) and self._has_back_to_search(available_actions):
            if self._is_search_action(action):
                last_query = self._extract_search_query(action) or self._last_search_query() or ""
                if last_query:
                    self.pending_search_query = next_broader_query(
                        observation.get("mission", "") or observation["text"]["long_term_context"],
                        self.failed_search_queries + [last_query],
                        max(self.zero_result_streak, self.search_retry_count, 1),
                    )
                return "click[back to search]", "zero_results_back_to_search"

        if self._is_results_page(observation, available_actions):
            item_click = self._first_product_click(available_actions)
            if item_click and (
                self._is_search_action(action)
                or action.lower() in {"click[back to search]", "click[next >]"}
            ):
                return item_click, "results_page_prefer_product_click"

        return action, method

    @staticmethod
    def _extract_action_candidate(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""

        action_line = re.search(r"Action:\s*(search\[[^\]]*\]|click\[[^\]]*\])", raw, flags=re.IGNORECASE)
        if action_line:
            return action_line.group(1).strip()

        # Extract first explicit action anywhere in output.
        match = re.search(r"(search\[[^\]]*\]|click\[[^\]]*\])", raw, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Otherwise first line is invalid candidate.
        return raw.splitlines()[0].strip()

    @staticmethod
    def _extract_thought_and_action(text: str) -> tuple[str, str]:
        raw = (text or "").strip()
        action_match = re.search(r"Action:\s*(search\[[^\]]*\]|click\[[^\]]*\])", raw, flags=re.IGNORECASE)
        thought_match = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", raw, flags=re.IGNORECASE | re.DOTALL)

        reasoning_text = thought_match.group(1).strip() if thought_match else ""
        action_text = action_match.group(1).strip() if action_match else WebShopReactAgent._extract_action_candidate(raw)
        return reasoning_text, action_text

    @staticmethod
    def _is_search_action(action: str) -> bool:
        return bool(re.fullmatch(r"search\[[^\]]*\]", (action or "").strip(), flags=re.IGNORECASE))

    @staticmethod
    def _normalize_search_action(action: str, fallback_query: str) -> str:
        match = re.fullmatch(r"search\[(.*)\]", (action or "").strip(), flags=re.IGNORECASE)
        if match is None:
            return f"search[{fallback_query}]"

        content = match.group(1).strip()
        if not content or content.lower() in {"<keywords>", "keywords", "query", "<query>"}:
            return f"search[{fallback_query}]"

        return f"search[{content}]"

    @staticmethod
    def _extract_search_query(action: str) -> str | None:
        match = re.fullmatch(r"search\[(.*)\]", (action or "").strip(), flags=re.IGNORECASE)
        return None if match is None else match.group(1).strip()

    @staticmethod
    def _is_zero_results_page(observation: dict) -> bool:
        text = observation["text"]["long_term_context"].lower()
        return "total results: 0" in text

    def _is_search_page(self, observation: dict, available_actions: list[str]) -> bool:
        text = observation["text"]["long_term_context"].lower()
        if self._is_results_page(observation, available_actions):
            return False
        if "back to search" in text:
            return False
        if "total results:" in text:
            return False
        if self._first_product_click(available_actions):
            return False
        return "[sep] search" in text or text.rstrip().endswith("search")

    @staticmethod
    def _has_back_to_search(available_actions: list[str]) -> bool:
        return "click[back to search]" in [a.lower() for a in available_actions]

    @staticmethod
    def _first_product_click(available_actions: list[str]) -> str | None:
        for action in available_actions:
            lowered = action.lower()
            if lowered.startswith("click[item - "):
                return action
            if lowered.startswith("click[") and lowered.endswith("]"):
                target = action[6:-1].strip()
                if ASIN_RE.fullmatch(target):
                    return action
        return None

    def _is_results_page(self, observation: dict, available_actions: list[str]) -> bool:
        text = observation["text"]["long_term_context"].lower()
        return bool(self._first_product_click(available_actions)) or "total results:" in text

    def _is_back_search_loop(self) -> bool:
        if len(self.history) < 4:
            return False
        a = [h["action"].lower() for h in self.history[-4:]]
        return (
            a[0].startswith("search[")
            and a[1] == "click[back to search]"
            and a[2] == a[0]
            and a[3] == "click[back to search]"
        )

    def _last_search_query(self) -> str | None:
        for item in reversed(self.history):
            query = self._extract_search_query(item.get("action", ""))
            if query:
                return query
        return None

    def _format_recent_history(self, max_steps: int = 4) -> str:
        if not self.history:
            return "[Search]\n\n"
        lines: list[str] = []
        for item in self.history[-max_steps:]:
            reasoning = (item.get("reasoning") or "").strip()
            action = (item.get("action") or "").strip()
            observation = (item.get("observation") or "").strip()
            if reasoning:
                lines.append(f"Thought: {reasoning}")
            if action:
                lines.append(f"Action: {action}")
            if observation:
                lines.append(f"Observation:\n{observation}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def trace_to_record(prefix: str, trace: MonitoringResponse) -> dict:
        trajectory = trace.prompt_monitor_prob_trajectory or trace.p_hack_trajectory
        return {
            f"{prefix}_text": trace.text,
            f"{prefix}_entropy_mean": trace.entropy_mean,
            f"{prefix}_entropy_max": trace.entropy_max,
            f"{prefix}_token_entropy": trace.token_entropy,
            f"{prefix}_p_hack": trace.p_hack,
            f"{prefix}_p_hack_trajectory": trace.p_hack_trajectory,
            f"{prefix}_prompt_monitor_prob_so_far": trace.prompt_monitor_prob_so_far,
            f"{prefix}_prompt_monitor_prob_trajectory": trace.prompt_monitor_prob_trajectory,
            f"{prefix}_p_hack_late_change": late_change(trajectory),
            f"{prefix}_p_hack_late_slope": late_slope(trajectory),
            f"{prefix}_entropy_late_change": late_change(trace.token_entropy),
            f"{prefix}_entropy_late_slope": late_slope(trace.token_entropy),
        }
