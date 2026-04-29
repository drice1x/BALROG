from __future__ import annotations

import copy
import re
from difflib import get_close_matches
from typing import Any

from balrog.agents.base import BaseAgent


class ReactPublicAgent(BaseAgent):
    """
    ReAct-style public baseline agent adapted for local vLLM inference.

    Supports:
    - ALFWorld admissible-command interface
    - WebShop text interface with click[...] and search[...]
    - monitor/entropy logging if the client returns them

    Output format:
        THOUGHT: ...
        ACTION: ...
    """

    THOUGHT_RE = re.compile(
        r"THOUGHT\s*:\s*(.*?)(?=(?:\nACTION\s*:)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    ACTION_RE = re.compile(
        r"ACTION\s*:\s*(.*)",
        re.IGNORECASE | re.DOTALL,
    )

    LEAK_MARKERS = [
        "user",
        "assistant",
        "system",
        "observation:",
        "previous action:",
        "thought:",
        "action:",
        "<thought>",
        "<action>",
        "end of response",
        "endoftext",
        "finish",
    ]

    def __init__(self, client_factory, prompt_builder, config=None):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.config = config
        self.last_trace: dict[str, Any] | None = None

        self.max_tokens = 64
        self.temperature = 0.0
        self.use_fewshot = True

        if config is not None and hasattr(config, "agent"):
            self.max_tokens = int(getattr(config.agent, "action_max_tokens", self.max_tokens))
            self.temperature = float(getattr(config.agent, "action_temperature", self.temperature))
            self.use_fewshot = bool(getattr(config.agent, "use_fewshot", self.use_fewshot))

    def reset(self):
        super().reset()
        self.last_trace = None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)
        messages = copy.deepcopy(self.prompt_builder.get_prompt())

        action_spec = self._get_action_spec(obs)
        action_prompt = self._build_action_prompt(obs, action_spec)

        if messages and getattr(messages[-1], "role", None) == "user":
            messages[-1].content += "\n\n" + action_prompt

        response = self.client.generate(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        raw_text = getattr(response, "completion", "") or ""
        thought = self._extract_thought(raw_text)
        raw_action = self._extract_action(raw_text)

        parsed_action, projection_meta = self._project_to_valid_action(
            raw_action=raw_action,
            action_spec=action_spec,
        )

        self.last_trace = {
            "raw_completion": raw_text,
            "thought": thought,
            "raw_action_extracted": raw_action,
            "parsed_action": parsed_action,
            "action_spec": action_spec,
            "projection_meta": projection_meta,
            "token_entropy": getattr(response, "token_entropy", []),
            "entropy_mean": getattr(response, "entropy_mean", None),
            "entropy_max": getattr(response, "entropy_max", None),
            "monitor": getattr(response, "monitor", None),
            "monitor_latest": getattr(response, "monitor_latest", None),
            "p_hack": getattr(response, "p_hack", None),
            "p_hack_trajectory": getattr(response, "p_hack_trajectory", []),
            "prompt_monitor_prob_so_far": getattr(response, "prompt_monitor_prob_so_far", None),
            "prompt_monitor_prob_trajectory": getattr(response, "prompt_monitor_prob_trajectory", []),
        }

        response.completion = parsed_action
        response.reasoning = thought
        return response

    def pop_last_trace(self):
        trace = self.last_trace
        self.last_trace = None
        return trace

    # -------------------------------------------------
    # Prompt building
    # -------------------------------------------------

    def _build_action_prompt(self, obs, action_spec: dict[str, Any]) -> str:
        obs_text = self._obs_to_text(obs).strip()
        env_family = action_spec.get("env_family")

        if env_family == "alfworld":
            valid_actions = action_spec.get("valid_actions", [])
            valid_block = "\n".join(f"- {a}" for a in valid_actions[:80])

            examples = ""
            if self.use_fewshot:
                examples = (
                    "Example 1:\n"
                    "Observation: You see a fridge and a table.\n"
                    "Valid actions:\n"
                    "- look\n"
                    "- go to fridge 1\n"
                    "- go to table 1\n"
                    "- open fridge 1\n\n"
                    "THOUGHT: The fridge is likely relevant.\n"
                    "ACTION: go to fridge 1\n\n"
                    "Example 2:\n"
                    "Observation: You are at the fridge and it is closed.\n"
                    "Valid actions:\n"
                    "- look\n"
                    "- open fridge 1\n"
                    "- go to table 1\n\n"
                    "THOUGHT: I should open the fridge first.\n"
                    "ACTION: open fridge 1\n\n"
                )

            return (
                "You are an ALFWorld agent.\n\n"
                f"{examples}"
                "Current observation:\n"
                f"{obs_text}\n\n"
                "Valid actions:\n"
                f"{valid_block}\n\n"
                "Rules:\n"
                "1. Think briefly.\n"
                "2. Choose EXACTLY ONE valid action from the list.\n"
                "3. Copy the action EXACTLY as written.\n"
                "4. Do NOT invent new actions.\n"
                "5. Output exactly:\n"
                "THOUGHT: <short thought>\n"
                "ACTION: <exact valid action>\n"
            )

        if env_family == "webshop":
            click_actions = action_spec.get("click_actions", [])
            allow_search = action_spec.get("allow_search", False)
            click_block = "\n".join(f"- {a}" for a in click_actions[:80]) if click_actions else "- none"

            search_line = (
                "You MAY also use a search action of the exact form search[keywords].\n"
                if allow_search else
                "Search is NOT available right now.\n"
            )

            examples = ""
            if self.use_fewshot:
                examples = (
                    "Example:\n"
                    "Observation: WebShop page with instruction and a search button.\n"
                    "Available click actions:\n"
                    "- click[search]\n\n"
                    "THOUGHT: I should search for the target item.\n"
                    "ACTION: search[wireless mouse ergonomic]\n\n"
                )

            return (
                "You are a WebShop agent.\n\n"
                f"{examples}"
                "Current observation:\n"
                f"{obs_text}\n\n"
                "Available click actions:\n"
                f"{click_block}\n\n"
                f"{search_line}\n"
                "Rules:\n"
                "1. Think briefly.\n"
                "2. Output exactly one action.\n"
                "3. Valid action forms are either click[exact_clickable] or search[keywords].\n"
                "4. For click actions, the clickable value must match exactly.\n"
                "5. Do NOT invent unavailable click targets.\n"
                "6. Output exactly:\n"
                "THOUGHT: <short thought>\n"
                "ACTION: <one valid action>\n"
            )

        return (
            "You are an agent.\n\n"
            "Current observation:\n"
            f"{obs_text}\n\n"
            "Output exactly:\n"
            "THOUGHT: <short thought>\n"
            "ACTION: <one action>\n"
        )

    # -------------------------------------------------
    # Action spec
    # -------------------------------------------------

    def _get_action_spec(self, obs) -> dict[str, Any]:
        obs_text = self._obs_to_text(obs).lower()

        if isinstance(obs, dict) and "admissible_commands" in obs:
            admissible = [
                str(a).strip().lower()
                for a in obs.get("admissible_commands", [])
                if str(a).strip()
            ]
            default_action = "look" if "look" in admissible else (admissible[0] if admissible else "look")
            return {
                "env_family": "alfworld",
                "valid_actions": admissible,
                "default_action": default_action,
            }

        if isinstance(obs, dict) and "webshop_clickables" in obs:
            clickables = [str(a).strip() for a in obs.get("webshop_clickables", []) if str(a).strip()]
            click_actions = [f"click[{c}]" for c in clickables]
            allow_search = bool(obs.get("webshop_has_search_bar", False))
            default_action = click_actions[0] if click_actions else ("search[]" if allow_search else "click[search]")
            return {
                "env_family": "webshop",
                "click_actions": click_actions,
                "allow_search": allow_search,
                "default_action": default_action,
            }

        return {
            "env_family": "unknown",
            "default_action": "look",
        }

    # -------------------------------------------------
    # Parsing / projection
    # -------------------------------------------------

    def _obs_to_text(self, obs: Any) -> str:
        if obs is None:
            return ""

        if isinstance(obs, str):
            return obs

        if isinstance(obs, dict):
            chunks = []
            text_part = obs.get("text")
            if isinstance(text_part, dict):
                for v in text_part.values():
                    if isinstance(v, str):
                        chunks.append(v)
            mission = obs.get("mission")
            if isinstance(mission, str):
                chunks.append(mission)
            for k, v in obs.items():
                if isinstance(v, str) and k != "mission":
                    chunks.append(v)
            return "\n".join(chunks)

        return str(obs)

    def _extract_thought(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        m = self.THOUGHT_RE.search(raw_text)
        if not m:
            return ""
        return re.sub(r"\s+", " ", m.group(1)).strip()

    def _extract_action(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        raw_text = raw_text.strip()
        m = self.ACTION_RE.search(raw_text)
        if m:
            raw_text = m.group(1).strip()

        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not lines:
            return ""

        text = lines[0].strip()

        lowered = text.lower()
        cut_idx = None
        for marker in self.LEAK_MARKERS:
            idx = lowered.find(marker)
            if idx > 0:
                cut_idx = idx if cut_idx is None else min(cut_idx, idx)

        if cut_idx is not None:
            text = text[:cut_idx].strip()

        return text

    def _project_to_valid_action(
        self,
        raw_action: str,
        action_spec: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        env_family = action_spec.get("env_family")
        default_action = action_spec.get("default_action", "look")

        meta = {
            "used_exact_match": False,
            "used_fuzzy_match": False,
            "used_search_action": False,
            "used_fallback": False,
            "raw_action": raw_action,
        }

        cleaned = (raw_action or "").strip()

        if env_family == "alfworld":
            valid_actions = [a.strip().lower() for a in action_spec.get("valid_actions", [])]
            action = cleaned.lower()

            if action in valid_actions:
                meta["used_exact_match"] = True
                return action, meta

            action2 = re.sub(r"[\"'`.,;:!?]+$", "", action).strip()
            if action2 in valid_actions:
                meta["used_exact_match"] = True
                return action2, meta

            match = get_close_matches(action2, valid_actions, n=1, cutoff=0.6)
            if match:
                meta["used_fuzzy_match"] = True
                return match[0], meta

            meta["used_fallback"] = True
            return default_action, meta

        if env_family == "webshop":
            click_actions = action_spec.get("click_actions", [])
            allow_search = bool(action_spec.get("allow_search", False))
            action = cleaned

            if action in click_actions:
                meta["used_exact_match"] = True
                return action, meta

            m = re.fullmatch(r"click\[(.*)\]", action)
            if m:
                target = m.group(1).strip()
                possible = [a for a in click_actions if a.lower() == f"click[{target.lower()}]"]
                if possible:
                    meta["used_exact_match"] = True
                    return possible[0], meta

                match = get_close_matches(
                    f"click[{target}]".lower(),
                    [a.lower() for a in click_actions],
                    n=1,
                    cutoff=0.6,
                )
                if match:
                    idx = [a.lower() for a in click_actions].index(match[0])
                    meta["used_fuzzy_match"] = True
                    return click_actions[idx], meta

            if allow_search:
                m = re.fullmatch(r"search\[(.*)\]", action, flags=re.DOTALL)
                if m:
                    query = m.group(1).strip()
                    meta["used_search_action"] = True
                    return f"search[{query}]", meta

            meta["used_fallback"] = True
            return default_action, meta

        if cleaned:
            return cleaned, meta

        meta["used_fallback"] = True
        return default_action, meta