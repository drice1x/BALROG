from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Any

from balrog.agents.base import BaseAgent


class CustomMonitorAgent(BaseAgent):
    """
    BALROG-style chain-of-thought agent with monitoring.

    Design goals:
    - keep the same high-level control flow as BALROG's ChainOfThoughtAgent
    - enforce a very clear REASONING / ACTION split
    - normalize actions into environment-valid commands
    - log entropy / monitor traces for analysis
    """

    ACTION_RE = re.compile(r"ACTION\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)
    REASONING_RE = re.compile(
        r"REASONING\s*:\s*(.*?)(?=\nACTION\s*:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    LEAK_MARKERS = [
        "user",
        "assistant",
        "system",
        "observation:",
        "previous action:",
        "previous plan:",
        "reasoning:",
        "action:",
        "<reasoning>",
        "<one action>",
        "end of response",
        "endoftext",
        "endoftxt",
        "finish",
    ]

    def __init__(self, client_factory, prompt_builder, config=None):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.last_trace: dict[str, Any] | None = None
        self.remember_cot = False
        if config is not None and hasattr(config, "agent"):
            self.remember_cot = bool(getattr(config.agent, "remember_cot", False))

    def reset(self):
        super().reset()
        self.last_trace = None

    def act(self, obs, prev_action=None):
        # Same outer flow as BALROG CoT agent
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        action_spec = self._get_action_spec(obs)
        self._append_action_instruction(messages, action_spec)

        response = self.client.generate(messages)

        raw_text = getattr(response, "completion", "") or ""
        reasoning = self._extract_reasoning(raw_text)
        raw_action = self._extract_action_text(raw_text)

        normalized_action, normalization_meta = self._normalize_action(
            raw_action=raw_action,
            action_spec=action_spec,
        )
        final_action = self._finalize_action(
            action=normalized_action,
            default_action=action_spec["default_action"],
            max_chars=action_spec["max_action_chars"],
        )

        # Optional CoT memory, like BALROG CoT agent
        if self.remember_cot and reasoning:
            self.prompt_builder.update_reasoning(reasoning)

        self.last_trace = {
            "raw_completion": raw_text,
            "raw_completion_empty": (raw_text.strip() == ""),
            "raw_action_extracted": raw_action,
            "action_empty_after_parsing": (raw_action == ""),
            "parsed_action": final_action,
            "reasoning": reasoning,
            "action_spec": action_spec,
            "normalization_meta": normalization_meta,
            "token_entropy": getattr(response, "token_entropy", []),
            "entropy_mean": getattr(response, "entropy_mean", None),
            "entropy_max": getattr(response, "entropy_max", None),
            "monitor": getattr(response, "monitor", None),
            "monitor_latest": getattr(response, "monitor_latest", None),
            "p_hack": getattr(response, "p_hack", None),
            "p_hack_trajectory": getattr(response, "p_hack_trajectory", []),
            "prompt_monitor_prob_so_far": getattr(
                response, "prompt_monitor_prob_so_far", None
            ),
            "prompt_monitor_prob_trajectory": getattr(
                response, "prompt_monitor_prob_trajectory", []
            ),
            "raw_response": getattr(response, "raw_response", None),
        }

        response.completion = final_action
        response.reasoning = reasoning
        return response

    def pop_last_trace(self):
        trace = self.last_trace
        self.last_trace = None
        return trace

    # ----------------------------
    # Parsing
    # ----------------------------

    def _extract_reasoning(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        match = self.REASONING_RE.search(raw_text)
        if not match:
            return ""
        return match.group(1).strip()

    def _extract_action_text(self, raw_text: str) -> str:
        """
        Strictly extract the action candidate.

        Rules:
        - prefer explicit ACTION:
        - keep only first line
        - lowercase + collapse whitespace
        - cut off prompt leakage markers
        """
        if not raw_text:
            return ""

        match = self.ACTION_RE.search(raw_text)
        text = match.group(1).strip() if match else raw_text.strip()

        if not text:
            return ""

        lines = text.splitlines()
        if not lines:
            return ""

        text = lines[0].strip()
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text).strip().lower()

        cut_idx = None
        for marker in self.LEAK_MARKERS:
            idx = text.find(marker)
            if idx > 0:
                cut_idx = idx if cut_idx is None else min(cut_idx, idx)

        if cut_idx is not None:
            text = text[:cut_idx].strip()

        return text

    def _finalize_action(self, action: str, default_action: str, max_chars: int) -> str:
        """
        Final safety pass before handing action to environment.
        """
        action = (action or "").splitlines()[0].strip().lower()
        action = re.sub(r"\s+", " ", action)

        for marker in self.LEAK_MARKERS:
            idx = action.find(marker)
            if idx > 0:
                action = action[:idx].strip()

        action = action[:max_chars].strip()

        if not action:
            return default_action

        return action

    # ----------------------------
    # Environment-specific action specs
    # ----------------------------

    def _get_action_spec(self, obs) -> dict[str, Any]:
        obs_text = self._obs_to_text(obs).lower()

        # TextWorld: conservative command set first.
        if any(key in obs_text for key in ["textworld", "coin", "treasure", "cooking", "goal:"]):
            return {
                "env_family": "textworld",
                "mode": "templated",
                "valid_actions": [
                    "goal",
                    "look",
                    "inventory",
                    "go north",
                    "go east",
                    "go south",
                    "go west",
                    "take coin",
                ],
                "default_action": "look",
                "max_action_chars": 64,
            }

        # BabyAI
        if "babyai" in obs_text or "mission" in obs_text:
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
        if any(key in obs_text for key in ["minihack", "nethack", "corridor", "maze", "quest"]):
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
                "default_action": "search",
                "max_action_chars": 16,
            }

        return {
            "env_family": "unknown",
            "mode": "free_text",
            "valid_actions": [],
            "default_action": "look",
            "max_action_chars": 32,
        }

    def _obs_to_text(self, obs) -> str:
        if obs is None:
            return ""

        if isinstance(obs, str):
            return obs

        if isinstance(obs, dict):
            chunks: list[str] = []
            for value in obs.values():
                if isinstance(value, str):
                    chunks.append(value)
                elif isinstance(value, dict):
                    for inner_value in value.values():
                        if isinstance(inner_value, str):
                            chunks.append(inner_value)
            return "\n".join(chunks)

        return str(obs)

    def _append_action_instruction(self, messages, action_spec: dict[str, Any]) -> None:
        if not messages or getattr(messages[-1], "role", None) != "user":
            return

        valid = "\n".join(f"- {a}" for a in action_spec["valid_actions"])

        if action_spec["mode"] in {"closed", "templated"}:
            instruction = (
                "\n\nThink step by step very briefly.\n"
                "Then output exactly one valid action.\n\n"
                "Valid actions / command forms:\n"
                f"{valid}\n\n"
                "Output exactly this format and nothing else:\n"
                "REASONING: <brief reasoning>\n"
                "ACTION: <one valid action>\n\n"
                "Rules:\n"
                "- The ACTION line must contain only the action.\n"
                "- The ACTION line must not be empty.\n"
                "- Do not include 'user' or 'assistant'.\n"
                "- Do not output multiple actions.\n"
                "- Do not repeat the prompt.\n"
                "- Keep reasoning short.\n"
            )
        else:
            instruction = (
                "\n\nThink step by step very briefly.\n"
                "Then output exactly this format:\n"
                "REASONING: <brief reasoning>\n"
                "ACTION: <one action>\n"
            )

        messages[-1].content += instruction

    # ----------------------------
    # Normalization
    # ----------------------------

    def _normalize_action(
        self,
        raw_action: str,
        action_spec: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        mode = action_spec["mode"]
        default_action = action_spec["default_action"]
        valid_actions = action_spec["valid_actions"]

        meta = {
            "mode": mode,
            "used_fallback": False,
            "used_fuzzy_match": False,
            "was_changed": False,
        }

        if mode == "free_text":
            cleaned = raw_action.strip()
            if not cleaned:
                meta["used_fallback"] = True
                return default_action, meta
            return cleaned, meta

        if action_spec["env_family"] == "textworld":
            normalized = self._normalize_textworld_action(raw_action, default_action)
            if normalized != raw_action:
                meta["was_changed"] = True
            if normalized == default_action and raw_action != default_action:
                meta["used_fallback"] = True
            return normalized, meta

        valid_norm = {a.lower(): a for a in valid_actions}

        if raw_action in valid_norm:
            return valid_norm[raw_action], meta

        match = get_close_matches(raw_action, list(valid_norm.keys()), n=1, cutoff=0.8)
        if match:
            meta["used_fuzzy_match"] = True
            meta["was_changed"] = True
            return valid_norm[match[0]], meta

        meta["used_fallback"] = True
        meta["was_changed"] = True
        return default_action, meta

    def _normalize_textworld_action(self, action: str, default_action: str) -> str:
        action = self._extract_action_text(action)

        if not action:
            return default_action

        # reject obvious non-actions
        banned_substrings = [
            "reasoning",
            "action",
            "then",
            "#",
            "<",
            ">",
        ]
        if any(b in action for b in banned_substrings):
            return default_action

        if any(ch.isdigit() for ch in action):
            return default_action

        valid_order = [
            "go north",
            "go east",
            "go south",
            "go west",
            "take coin",
            "goal",
            "look",
            "inventory",
        ]

        for candidate in valid_order:
            if action == candidate:
                return candidate

        return default_action