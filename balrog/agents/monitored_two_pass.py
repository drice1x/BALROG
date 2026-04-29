from __future__ import annotations

import copy
import re
from difflib import get_close_matches
from typing import Any

from balrog.agents.base import BaseAgent


class MonitoredTwoPassAgent(BaseAgent):
    """
    Two-pass monitored agent.

    Pass 1: reasoning only
      - collects entropy and monitor trajectories
      - does NOT send its text to the environment

    Pass 2: action only
      - outputs exactly one valid action
      - strongly normalized to the environment action space
    """

    REASONING_RE = re.compile(
        r"REASONING\s*:\s*(.*?)(?=\Z)",
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
        self.config = config
        self.last_trace: dict[str, Any] | None = None

        self.remember_cot = False
        self.reasoning_max_tokens = 16
        self.action_max_tokens = 8
        self.reasoning_temperature = 0.7
        self.action_temperature = 0.0

        if config is not None and hasattr(config, "agent"):
            self.remember_cot = bool(getattr(config.agent, "remember_cot", False))
            self.reasoning_max_tokens = int(
                getattr(config.agent, "reasoning_max_tokens", self.reasoning_max_tokens)
            )
            self.action_max_tokens = int(
                getattr(config.agent, "action_max_tokens", self.action_max_tokens)
            )
            self.reasoning_temperature = float(
                getattr(config.agent, "reasoning_temperature", self.reasoning_temperature)
            )
            self.action_temperature = float(
                getattr(config.agent, "action_temperature", self.action_temperature)
            )

    def reset(self):
        super().reset()
        self.last_trace = None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)
        base_messages = self.prompt_builder.get_prompt()

        action_spec = self._get_action_spec(obs)

        # -------------------------
        # Pass 1: reasoning
        # -------------------------
        reasoning_messages = copy.deepcopy(base_messages)
        reasoning_instruction = (
            "\n\nThink briefly about the next best action. "
            "Do not output the final action yet. "
            "Keep reasoning short."
        )

        if reasoning_messages and getattr(reasoning_messages[-1], "role", None) == "user":
            reasoning_messages[-1].content += reasoning_instruction

        reasoning_resp = self.client.generate(
            reasoning_messages,
            max_tokens=self.reasoning_max_tokens,
            temperature=self.reasoning_temperature,
        )
        reasoning_text = getattr(reasoning_resp, "completion", "") or ""

        # -------------------------
        # Pass 2: action
        # -------------------------
        action_messages = copy.deepcopy(base_messages)
        action_prompt = self._build_action_prompt(obs, reasoning_text, action_spec)

        if action_messages and getattr(action_messages[-1], "role", None) == "user":
            action_messages[-1].content += "\n\n" + action_prompt

        action_resp = self.client.generate(
            action_messages,
            max_tokens=self.action_max_tokens,
            temperature=self.action_temperature,
        )

        raw_action_text = getattr(action_resp, "completion", "") or ""
        extracted_action = self._extract_action(raw_action_text)

        projected_action, projection_meta = self._project_to_valid_action(
            extracted_action,
            action_spec.get("valid_actions", []),
            action_spec.get("default_action", "look"),
        )

        self.last_trace = {
            "reasoning_raw_completion": reasoning_text,
            "action_raw_completion": raw_action_text,
            "action_raw_extracted": extracted_action,
            "parsed_action": projected_action,
            "action_spec": action_spec,
            "projection_meta": projection_meta,
            "reasoning_token_entropy": getattr(reasoning_resp, "token_entropy", []),
            "reasoning_entropy_mean": getattr(reasoning_resp, "entropy_mean", None),
            "reasoning_entropy_max": getattr(reasoning_resp, "entropy_max", None),
            "reasoning_monitor": getattr(reasoning_resp, "monitor", None),
            "reasoning_monitor_latest": getattr(reasoning_resp, "monitor_latest", None),
            "reasoning_p_hack": getattr(reasoning_resp, "p_hack", None),
            "reasoning_p_hack_trajectory": getattr(reasoning_resp, "p_hack_trajectory", []),
            "reasoning_prompt_monitor_prob_so_far": getattr(
                reasoning_resp, "prompt_monitor_prob_so_far", None
            ),
            "reasoning_prompt_monitor_prob_trajectory": getattr(
                reasoning_resp, "prompt_monitor_prob_trajectory", []
            ),
            "action_token_entropy": getattr(action_resp, "token_entropy", []),
            "action_entropy_mean": getattr(action_resp, "entropy_mean", None),
            "action_entropy_max": getattr(action_resp, "entropy_max", None),
            "action_monitor": getattr(action_resp, "monitor", None),
            "action_monitor_latest": getattr(action_resp, "monitor_latest", None),
            "action_p_hack": getattr(action_resp, "p_hack", None),
            "action_p_hack_trajectory": getattr(action_resp, "p_hack_trajectory", []),
            "action_prompt_monitor_prob_so_far": getattr(
                action_resp, "prompt_monitor_prob_so_far", None
            ),
            "action_prompt_monitor_prob_trajectory": getattr(
                action_resp, "prompt_monitor_prob_trajectory", []
            ),
            "ttc": {
                "reasoning_max_tokens": self.reasoning_max_tokens,
                "action_max_tokens": self.action_max_tokens,
                "reasoning_temperature": self.reasoning_temperature,
                "action_temperature": self.action_temperature,
            },
        }

        action_resp.completion = projected_action
        action_resp.reasoning = reasoning_text
        return action_resp

    def pop_last_trace(self):
        trace = self.last_trace
        self.last_trace = None
        return trace

    # ----------------------------
    # Prompting
    # ----------------------------

    def _build_action_prompt(self, obs, reasoning_text: str, action_spec: dict[str, Any]) -> str:
        obs_text = self._obs_to_text(obs).strip()
        valid_actions = action_spec.get("valid_actions", [])

        if action_spec["mode"] in {"closed", "admissible_list"} and valid_actions:
            valid_block = "\n".join(f"- {a}" for a in valid_actions)

            return (
                "You are selecting the next environment action.\n\n"
                "Current observation:\n"
                f"{obs_text}\n\n"
                "Your previous reasoning:\n"
                f"{reasoning_text.strip()}\n\n"
                "You must choose EXACTLY ONE action from the valid action list below.\n"
                "Valid actions:\n"
                f"{valid_block}\n\n"
                "CRITICAL RULES:\n"
                "1. You MUST copy the action EXACTLY as written.\n"
                "2. Do NOT change numbers, words, or formatting.\n"
                "3. Do NOT invent new actions.\n"
                "4. Do NOT explain your answer.\n"
                "5. Do NOT add 'ACTION:'.\n"
                "6. Output ONLY the action string.\n\n"
                "Your answer must be EXACTLY one of the actions above."
            )

        return (
            "You are selecting the next environment action.\n\n"
            "Current observation:\n"
            f"{obs_text}\n\n"
            "Your previous reasoning:\n"
            f"{reasoning_text.strip()}\n\n"
            "Return exactly one action and nothing else."
        )

    # ----------------------------
    # Parsing / projection
    # ----------------------------

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

    def _extract_action(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        raw_text = raw_text.strip()

        match = self.ACTION_RE.search(raw_text)
        if match:
            raw_text = match.group(1).strip()

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
    ) -> tuple[str, dict]:
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

    def _normalize_babyai_action(self, action: str, default_action: str) -> str:
        action = self._extract_action(action)
        if not action:
            return default_action

        mapping = {
            "turn left": "left",
            "left": "left",
            "turn right": "right",
            "right": "right",
            "go forward": "forward",
            "move forward": "forward",
            "forward": "forward",
            "pick up": "pickup",
            "pickup": "pickup",
            "drop": "drop",
            "toggle": "toggle",
            "done": "done",
        }
        return mapping.get(action, default_action)

    # ----------------------------
    # Action specs
    # ----------------------------

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

        # TextWorld
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
        if isinstance(obs, dict):
            if "mission" in obs:
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

            text_part = obs.get("text")
            if isinstance(text_part, dict):
                joined = " | ".join(
                    str(v) for v in text_part.values() if isinstance(v, str)
                ).lower()
                if any(k in joined for k in ["mission", "you see", "goal", "object", "forward", "left", "right"]):
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

        if any(key in obs_text for key in ["babyai", "mission", "you see", "goal object"]):
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