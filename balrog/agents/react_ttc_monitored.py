from __future__ import annotations

import copy
import re
from difflib import get_close_matches
from typing import Any

from balrog.agents.base import BaseAgent


class ReactTTCMonitoredAgent(BaseAgent):
    """
    TTC + Monitoring + Correct ALFWorld interface

    Design:
    - Pass 1: reasoning (TTC controlled)
    - Pass 2: action (strict admissible selection)
    - Uses ALFWorld admissible commands
    - Logs entropy + monitor signals

    This is your MAIN method for the paper.
    """

    ACTION_RE = re.compile(r"action\s*:\s*(.*)", re.IGNORECASE)

    def __init__(self, client_factory, prompt_builder, config=None):
        super().__init__(client_factory, prompt_builder)

        self.client = client_factory()
        self.config = config

        # TTC parameters
        self.reasoning_max_tokens = 16
        self.action_max_tokens = 8
        self.reasoning_temperature = 0.7
        self.action_temperature = 0.0

        if config is not None and hasattr(config, "agent"):
            self.reasoning_max_tokens = int(
                getattr(config.agent, "reasoning_max_tokens", 16)
            )
            self.action_max_tokens = int(
                getattr(config.agent, "action_max_tokens", 8)
            )
            self.reasoning_temperature = float(
                getattr(config.agent, "reasoning_temperature", 0.7)
            )
            self.action_temperature = float(
                getattr(config.agent, "action_temperature", 0.0)
            )

        self.last_trace = None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)
        base_messages = self.prompt_builder.get_prompt()

        action_spec = self._get_action_spec(obs)

        # =========================
        # PASS 1 — REASONING (TTC)
        # =========================
        reasoning_msgs = copy.deepcopy(base_messages)

        if reasoning_msgs[-1].role == "user":
            reasoning_msgs[-1].content += (
                "\n\nThink step by step about the best next action. "
                "Keep reasoning short."
            )

        reasoning_resp = self.client.generate(
            reasoning_msgs,
            max_tokens=self.reasoning_max_tokens,
            temperature=self.reasoning_temperature,
        )

        reasoning_text = reasoning_resp.completion or ""

        # =========================
        # PASS 2 — ACTION
        # =========================
        action_msgs = copy.deepcopy(base_messages)

        action_prompt = self._build_action_prompt(
            obs,
            reasoning_text,
            action_spec,
        )

        if action_msgs[-1].role == "user":
            action_msgs[-1].content += "\n\n" + action_prompt

        action_resp = self.client.generate(
            action_msgs,
            max_tokens=self.action_max_tokens,
            temperature=self.action_temperature,
        )

        raw_action = self._extract_action(action_resp.completion or "")

        final_action = self._project_action(
            raw_action,
            action_spec["valid_actions"],
            action_spec["default_action"],
        )

        # =========================
        # LOGGING
        # =========================
        # =========================
        # LOGGING (critical fix)
        # =========================

        trace = {
            "reasoning": reasoning_text,
            "action_raw": raw_action,
            "action_final": final_action,
            "ttc": self.reasoning_max_tokens,

            # ---------- entropy ----------
            "reasoning_entropy_mean": getattr(reasoning_resp, "entropy_mean", None),
            "reasoning_entropy_max": getattr(reasoning_resp, "entropy_max", None),
            "reasoning_token_entropy": getattr(reasoning_resp, "token_entropy", []),

            "action_entropy_mean": getattr(action_resp, "entropy_mean", None),
            "action_entropy_max": getattr(action_resp, "entropy_max", None),
            "action_token_entropy": getattr(action_resp, "token_entropy", []),

            # ---------- monitor ----------
            "reasoning_p_hack": getattr(reasoning_resp, "p_hack", None),
            "reasoning_p_hack_trajectory": getattr(
                reasoning_resp,
                "p_hack_trajectory",
                [],
            ),
            "reasoning_prompt_monitor_prob_so_far": getattr(
                reasoning_resp,
                "prompt_monitor_prob_so_far",
                None,
            ),
            "reasoning_prompt_monitor_prob_trajectory": getattr(
                reasoning_resp,
                "prompt_monitor_prob_trajectory",
                [],
            ),

            "action_p_hack": getattr(action_resp, "p_hack", None),
            "action_p_hack_trajectory": getattr(
                action_resp,
                "p_hack_trajectory",
                [],
            ),
            "action_prompt_monitor_prob_so_far": getattr(
                action_resp,
                "prompt_monitor_prob_so_far",
                None,
            ),
            "action_prompt_monitor_prob_trajectory": getattr(
                action_resp,
                "prompt_monitor_prob_trajectory",
                [],
            ),
        }

        # Keep legacy compatibility
        self.last_trace = trace

        # CRITICAL:
        # attach trace directly to response object so evaluator writes it
        action_resp.agent_trace = trace

        # also flatten most important fields for robustness
        action_resp.reasoning_entropy_mean = trace["reasoning_entropy_mean"]
        action_resp.action_entropy_mean = trace["action_entropy_mean"]
        action_resp.reasoning_p_hack = trace["reasoning_p_hack"]
        action_resp.action_p_hack = trace["action_p_hack"]

        action_resp.completion = final_action
        action_resp.reasoning = reasoning_text

        return action_resp

    # =========================
    # ACTION INTERFACE
    # =========================

    def _get_action_spec(self, obs):
        admissible = [
            str(a).strip().lower()
            for a in obs.get("admissible_commands", [])
            if str(a).strip()
        ]

        default = "look" if "look" in admissible else admissible[0]

        return {
            "valid_actions": admissible,
            "default_action": default,
        }

    def _build_action_prompt(self, obs, reasoning, spec):
        valid = "\n".join(f"- {a}" for a in spec["valid_actions"][:50])

        return (
            "You must choose EXACTLY ONE action.\n\n"
            f"Reasoning:\n{reasoning}\n\n"
            "Valid actions:\n"
            f"{valid}\n\n"
            "Rules:\n"
            "- Copy action EXACTLY\n"
            "- No modifications\n"
            "- No explanation\n"
            "Return ONLY the action"
        )

    def _extract_action(self, text):
        if not text:
            return ""

        m = self.ACTION_RE.search(text)
        if m:
            return m.group(1).strip().lower()

        return text.strip().split("\n")[0].lower()

    def _project_action(self, action, valid, default):
        valid = [v.lower() for v in valid]

        if action in valid:
            return action

        match = get_close_matches(action, valid, n=1, cutoff=0.6)
        if match:
            return match[0]

        return default