from __future__ import annotations

import copy
import re
from difflib import get_close_matches
from typing import Any

from balrog.agents.base import BaseAgent


class ReactAdmissibleAgent(BaseAgent):
    """
    ReAct-style agent for text environments with a strict admissible-action interface.

    Design:
    - The model may output short reasoning ("Thought:")
    - The model must output exactly one valid action from the provided action list
    - We project imperfect outputs back onto the valid action set

    Best use:
    - ALFWorld
    - other closed-action or admissible-command environments
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

        self.max_tokens = 32
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
            raw_action,
            action_spec.get("valid_actions", []),
            action_spec.get("default_action", "look"),
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
        valid_actions = action_spec.get("valid_actions", [])
        valid_block = "\n".join(f"- {a}" for a in valid_actions)

        examples = ""
        if self.use_fewshot and action_spec.get("env_family") == "alfworld":
            examples = (
                "Examples:\n"
                "Observation: You see a fridge and an apple.\n"
                "Valid actions:\n"
                "- look\n"
                "- go to fridge 1\n"
                "- open fridge 1\n"
                "- go to table 1\n\n"
                "THOUGHT: The fridge is likely relevant.\n"
                "ACTION: go to fridge 1\n\n"
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
            "Your job is to choose the next action.\n"
            "You may think briefly, but you must choose exactly one valid action.\n\n"
            f"{examples}"
            "Current observation:\n"
            f"{obs_text}\n\n"
            "Valid actions:\n"
            f"{valid_block}\n\n"
            "CRITICAL RULES:\n"
            "1. The ACTION must be EXACTLY one action from the valid action list.\n"
            "2. Do NOT change spelling, numbers, or wording.\n"
            "3. Do NOT invent new actions.\n"
            "4. Keep THOUGHT short.\n"
            "5. Output exactly this format:\n"
            "THOUGHT: <short thought>\n"
            "ACTION: <exact valid action>\n"
        )

    # -------------------------------------------------
    # Action spec
    # -------------------------------------------------

    def _get_action_spec(self, obs) -> dict[str, Any]:
        obs_text = self._obs_to_text(obs).lower()

        # ALFWorld
        if isinstance(obs, dict) and "admissible_commands" in obs:
            admissible = [
                str(a).strip().lower()
                for a in obs.get("admissible_commands", [])
                if str(a).strip()
            ]
            default_action = "look" if "look" in admissible else (admissible[0] if admissible else "look")
            return {
                "env_family": "alfworld",
                "mode": "admissible_list",
                "valid_actions": admissible,
                "default_action": default_action,
                "max_action_chars": 128,
            }

        # BabyAI
        if isinstance(obs, dict) and ("mission" in obs or "babyai" in obs_text):
            return {
                "env_family": "babyai",
                "mode": "closed",
                "valid_actions": [
                    "left",
                    "right",
                    "forward",
                    "pickup",
                    "drop",
                    "toggle",
                    "done",
                ],
                "default_action": "forward",
                "max_action_chars": 16,
            }

        # MiniHack
        if any(key in obs_text for key in ["minihack", "nethack", "maze", "corridor", "quest"]):
            return {
                "env_family": "minihack",
                "mode": "closed",
                "valid_actions": [
                    "north",
                    "south",
                    "east",
                    "west",
                    "northeast",
                    "northwest",
                    "southeast",
                    "southwest",
                    "search",
                    "kick",
                    "open",
                    "eat",
                    "wait",
                ],
                "default_action": "wait",
                "max_action_chars": 16,
            }

        return {
            "env_family": "unknown",
            "mode": "free_text",
            "valid_actions": [],
            "default_action": "look",
            "max_action_chars": 64,
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

        text = lines[0].strip().lower()

        cut_idx = None
        for marker in self.LEAK_MARKERS:
            idx = text.find(marker)
            if idx > 0:
                cut_idx = idx if cut_idx is None else min(cut_idx, idx)

        if cut_idx is not None:
            text = text[:cut_idx].strip()

        return text

    def _project_to_valid_action(
        self,
        raw_action: str,
        valid_actions: list[str],
        default_action: str,
    ) -> tuple[str, dict[str, Any]]:
        raw_action = (raw_action or "").strip().lower()
        valid_actions = [a.strip().lower() for a in valid_actions]

        meta = {
            "used_exact_match": False,
            "used_fuzzy_match": False,
            "used_substring_match": False,
            "used_fallback": False,
            "raw_action": raw_action,
        }

        if not valid_actions:
            meta["used_fallback"] = True
            return default_action, meta

        if raw_action in valid_actions:
            meta["used_exact_match"] = True
            return raw_action, meta

        cleaned = re.sub(r"[\"'`.,;:!?]+$", "", raw_action).strip()
        if cleaned in valid_actions:
            meta["used_exact_match"] = True
            return cleaned, meta

        substring_matches = [a for a in valid_actions if raw_action and raw_action in a]
        if len(substring_matches) == 1:
            meta["used_substring_match"] = True
            return substring_matches[0], meta

        reverse_substring_matches = [a for a in valid_actions if a in raw_action]
        if len(reverse_substring_matches) == 1:
            meta["used_substring_match"] = True
            return reverse_substring_matches[0], meta

        match = get_close_matches(cleaned, valid_actions, n=1, cutoff=0.6)
        if match:
            meta["used_fuzzy_match"] = True
            return match[0], meta

        meta["used_fallback"] = True
        return default_action, meta