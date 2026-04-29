from __future__ import annotations

import re

from balrog.agents.base import BaseAgent


class CustomMonitorAgent(BaseAgent):
    ACTION_RE = re.compile(r"ACTION\\s*:\\s*(.*)", re.IGNORECASE | re.DOTALL)
    REASONING_RE = re.compile(
        r"REASONING\\s*:\\s*(.*?)(?=\\nACTION\\s*:|\\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, client_factory, prompt_builder):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.last_trace = None

    def reset(self):
        super().reset()
        self.last_trace = None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)
        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        instruction = (
            "\\n\\nYou are an agent acting in an environment. "
            "Output exactly one action.\\n"
            "Format:\\n"
            "REASONING: <brief reasoning>\\n"
            "ACTION: <one valid action>"
        )

        if messages and getattr(messages[-1], "role", None) == "user":
            messages[-1].content += instruction

        response = self.client.generate(messages)

        raw_text = response.completion or ""
        reasoning = ""
        action = raw_text.strip()

        m_reason = self.REASONING_RE.search(raw_text)
        if m_reason:
            reasoning = m_reason.group(1).strip()

        m_action = self.ACTION_RE.search(raw_text)
        if m_action:
            action = m_action.group(1).strip()

        self.last_trace = {
            "raw_completion": raw_text,
            "parsed_action": action,
            "reasoning": reasoning,
            "token_entropy": getattr(response, "token_entropy", []),
            "entropy_mean": getattr(response, "entropy_mean", None),
            "entropy_max": getattr(response, "entropy_max", None),
            "monitor": getattr(response, "monitor", None),
            "monitor_latest": getattr(response, "monitor_latest", None),
            "p_hack": getattr(response, "p_hack", None),
            "p_hack_trajectory": getattr(response, "p_hack_trajectory", []),
            "prompt_monitor_prob_so_far": getattr(response, "prompt_monitor_prob_so_far", None),
            "prompt_monitor_prob_trajectory": getattr(response, "prompt_monitor_prob_trajectory", []),
            "raw_response": getattr(response, "raw_response", None),
        }

        response.completion = action
        response.reasoning = reasoning
        return response

    def pop_last_trace(self):
        trace = self.last_trace
        self.last_trace = None
        return trace