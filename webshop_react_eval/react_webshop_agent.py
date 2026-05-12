from __future__ import annotations

from dataclasses import dataclass
import re

from monitoring_client import MonitoringClient, MonitoringResponse


REACT_WEBSHOP_FEWSHOT = """Webshop
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[Search]

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

Action: think[B078GWRC1J matches bright citrus and sensitive skin and is under the price limit.]
Observation: OK.

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

Action: think[I should choose the exact matching options before buying.]
Observation: OK.

Action: click[bright citrus]
Observation: You have clicked bright citrus.
Action: click[3 ounce (pack of 1)]
Observation: You have clicked 3 ounce (pack of 1).
Action: click[Buy Now]
"""


@dataclass
class AgentStep:
    action: str
    raw_action: str
    thought_text: str
    trace: MonitoringResponse


def _clean_instruction(instruction: str) -> str:
    text = (instruction or "").replace("[SEP]", " ")
    text = re.sub(r"(?i)\binstruction:?|webshop\b", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 /-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_task_query(instruction: str) -> str:
    text = _clean_instruction(instruction)
    stop = {"find", "me", "a", "an", "the", "with", "for", "and", "price", "lower", "than", "dollars"}
    tokens = [tok for tok in text.split() if tok.lower() not in stop]
    return " ".join(tokens[:10]) or text or "product"


def build_broad_search_query(instruction: str) -> str:
    text = _clean_instruction(instruction)
    text = re.sub(r"(?i)^find me\s+", "", text)
    text = re.sub(r"(?i)\band price\b.*$", " ", text)
    text = re.sub(r"(?i)\bprice lower than\b.*$", " ", text)
    text = re.sub(r"(?i)\b(with|and)\s+(size|color|colour|price)\b.*$", " ", text)

    head = re.split(r"(?i)\b(?:for|with)\b", text, maxsplit=1)[0].strip(" ,")
    if not head:
        head = text

    clauses = [clause.strip() for clause in head.split(",") if clause.strip()]
    candidate = clauses[-1] if clauses else head
    tokens = candidate.split()
    if len(tokens) > 3:
        candidate = " ".join(tokens[-3:])

    candidate = re.sub(r"(?i)\b(double sided|machine washable|printing technology|eco friendly|high quality|easy clean|easy use|long lasting|assembly required|fully assembled|ready use|non slip|wall mounted|height adjustable|stainless steel|solid wood|white item|gray item)\b", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" ,")
    if candidate:
        return candidate

    fallback_tokens = [
        tok for tok in text.split()
        if tok.lower() not in {"find", "me", "with", "for", "and", "price", "lower", "than", "dollars", "size", "color", "colour"}
    ]
    return " ".join(fallback_tokens[-3:]) or build_task_query(instruction)


ASIN_RE = re.compile(r"^click\[[A-Z0-9]{8,12}\]$", re.I)
SELECTED_OPTION_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _instruction_tokens(instruction: str) -> set[str]:
    text = _clean_instruction(instruction).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    stop = {
        "find", "me", "a", "an", "the", "with", "for", "and", "price", "lower", "than",
        "dollars", "size", "color", "colour", "x", "inch", "inches",
    }
    return {tok for tok in tokens if tok not in stop and len(tok) > 2}


def choose_best_asin_action(observation: dict) -> str:
    available_actions = observation.get("available_actions", []) or []
    asin_actions = [str(a) for a in available_actions if ASIN_RE.fullmatch(str(a))]
    if not asin_actions:
        return ""

    text = observation.get("text", {}).get("long_term_context", "") or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    titles: dict[str, str] = {}
    for idx, line in enumerate(lines[:-1]):
        asin = None
        match = re.fullmatch(r"\[?([A-Z0-9]{8,12})\]?", line, flags=re.I)
        if match:
            asin = match.group(1).upper()
        if not asin:
            continue
        title = lines[idx + 1] if idx + 1 < len(lines) else ""
        titles[asin] = title

    inst_tokens = _instruction_tokens(observation.get("mission", ""))
    best_action = asin_actions[0]
    best_score = -1
    for action in asin_actions:
        asin = action[6:-1].upper()
        title_tokens = set(re.findall(r"[a-z0-9]+", titles.get(asin, "").lower()))
        score = len(inst_tokens & title_tokens)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def selected_options(observation: dict) -> set[str]:
    text = observation.get("text", {}).get("long_term_context", "") or ""
    return {match.lower().strip() for match in SELECTED_OPTION_RE.findall(text)}


class ReactWebShopAgent:
    def __init__(self, client: MonitoringClient, max_tokens: int = 64, temperature: float = 0.0):
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history: list[dict[str, str]] = []

    def reset(self):
        self.history = []

    def act(self, observation: dict) -> AgentStep:
        prompt = self._build_prompt(observation)
        trace = self.client.complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["\n"],
        )
        action = self._extract_action(trace.text)
        action = self._normalize_action(action, observation)
        if not action:
            best_asin = choose_best_asin_action(observation)
            if best_asin:
                action = best_asin
            elif "click[Buy Now]" in (observation.get("available_actions", []) or []) and selected_options(observation):
                action = "click[Buy Now]"
            else:
                action = "search[" + build_broad_search_query(observation.get("mission", "")) + "]"
        action = self._finalize_action(action, observation)

        self.history.append(
            {
                "observation": observation["text"]["long_term_context"],
                "action": action,
            }
        )
        return AgentStep(action=action, raw_action=trace.text, thought_text="", trace=trace)

    def _build_prompt(self, observation: dict) -> str:
        return build_react_webshop_prompt(observation, self.history)

    def _format_history(self) -> str:
        if not self.history:
            return "[Search]\n\n"
        lines = []
        for item in self.history[-4:]:
            lines.append(f"Action: {item['action']}")
            lines.append(f"Observation:\n{item['observation']}")
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _extract_action(text: str) -> str:
        matches = re.findall(r"Action:\s*(search\[[^\]]*\]|click\[[^\]]*\]|think\[[^\]]*\])", text, flags=re.I)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"(search\[[^\]]*\]|click\[[^\]]*\]|think\[[^\]]*\])", text, flags=re.I)
        for candidate in reversed(matches):
            lowered = candidate.strip().lower()
            if lowered not in {"search[query]", "search[keywords]"}:
                return candidate.strip()
        return ""

    @staticmethod
    def _normalize_action(action: str, observation: dict) -> str:
        if not action:
            return ""
        instruction = observation.get("mission", "") or ""
        fallback_query = build_task_query(instruction)
        broad_query = build_broad_search_query(instruction)
        best_asin = choose_best_asin_action(observation)
        selected = selected_options(observation)
        available_actions = observation.get("available_actions", []) or []
        lowered = action.lower()
        if lowered.startswith("think["):
            return action
        if lowered in {"search[query]", "search[keywords]"}:
            return f"search[{fallback_query}]"
        if lowered == "search[3 ounce bright citrus deodorant sensitive skin]":
            return f"search[{fallback_query}]"
        if lowered.startswith("search[instruction:"):
            return f"search[{fallback_query}]"
        if lowered.startswith("search[") and best_asin:
            return best_asin
        if lowered.startswith("click["):
            button = action[6:-1].strip().lower()
            if button in selected and "click[Buy Now]" in available_actions:
                return "click[Buy Now]"
        if lowered.startswith("search["):
            query = action[7:-1].strip()
            query_tokens = query.split()
            too_specific = (
                len(query_tokens) > 5
                or re.search(r"(?i)\b(size|color|colour|price|dollars|lower|than|inch|inches)\b", query)
                or query.lower() == fallback_query.lower()
            )
            if too_specific:
                return f"search[{broad_query}]"
        return action

    @staticmethod
    def _finalize_action(action: str, observation: dict) -> str:
        if not action:
            return action
        available_actions = observation.get("available_actions", []) or []
        selected = selected_options(observation)
        if "click[Buy Now]" not in available_actions or not selected:
            return action
        lowered = action.lower()
        if lowered.startswith("click["):
            button = action[6:-1].strip().lower()
            if button in selected:
                return "click[Buy Now]"
        return action


def build_react_webshop_prompt(observation: dict, history: list[dict[str, str]] | None = None) -> str:
    history = history or []
    task_query = build_task_query(observation.get("mission", ""))
    broad_query = build_broad_search_query(observation.get("mission", ""))
    available_actions = observation.get("available_actions", []) or []
    has_asin_results = any(ASIN_RE.fullmatch(str(a)) for a in available_actions)
    result_note = (
        "Product results are visible. Do not search again. Click one product result.\n"
        if has_asin_results
        else ""
    )
    search_note = (
        f"On the first search, use a broad product query like: {broad_query}\n"
        "Do not include size, color, price, or every attribute in the first search.\n"
    )
    return (
        "Below is one example trajectory that demonstrates the format only.\n"
        "Do not copy the example product or query for the current task.\n\n"
        f"{REACT_WEBSHOP_FEWSHOT}\n"
        "Now solve the current task below.\n\n"
        "Webshop\n"
        f"Instruction:\n{observation.get('mission', '')}\n"
        f"{format_react_webshop_history(history)}"
        f"Observation:\n{observation['text']['long_term_context']}\n\n"
        f"Task-specific search hint: {task_query}\n\n"
        f"{search_note}"
        "Generate exactly one next action.\n"
        "Valid actions include search[query], click[option], or think[brief reasoning].\n"
        "Use terms from the current instruction, not from the example.\n"
        f"{result_note}"
        "Action:"
    )


def format_react_webshop_history(history: list[dict[str, str]] | None = None) -> str:
    history = history or []
    if not history:
        return "[Search]\n\n"
    lines = []
    for item in history[-4:]:
        lines.append(f"Action: {item['action']}")
        lines.append(f"Observation:\n{item['observation']}")
    return "\n".join(lines) + "\n\n"
