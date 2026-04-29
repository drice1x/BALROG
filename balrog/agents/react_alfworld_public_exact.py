from __future__ import annotations

import copy
from typing import Any

from balrog.agents.base import BaseAgent


class ReactAlfworldPublicExactAgent(BaseAgent):
    """
    Faithful ReAct-style agent for ALFWorld.

    Key properties:
    - single-pass generation
    - trajectory-style prompting
    - "think:" is treated as an action
    - NO action projection / correction
    - NO admissible action injection

    This is your *paper baseline*.
    """

    def __init__(self, client_factory, prompt_builder, config=None):
        super().__init__(client_factory, prompt_builder)
        self.client = client_factory()
        self.config = config

        self.max_tokens = 64
        self.temperature = 0.0

    def reset(self):
        super().reset()

    def act(self, obs, prev_action=None):
        # === standard trajectory update ===
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = copy.deepcopy(self.prompt_builder.get_prompt())

        # === ReAct instruction ===
        react_instruction = (
            "\n\nYou are an agent acting in an environment.\n"
            "At each step, you can think or act.\n\n"
            "Format:\n"
            "Thought: <reasoning>\n"
            "Action: <action>\n\n"
            "Rules:\n"
            "- You may use 'think:' to reason without acting\n"
            "- Otherwise output a valid environment action\n"
            "- Do NOT output anything else\n"
        )

        if messages and getattr(messages[-1], "role", None) == "user":
            messages[-1].content += react_instruction

        response = self.client.generate(
            messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        raw = (response.completion or "").strip()

        # === ReAct-style parsing ===
        action = self._extract_action(raw)

        # IMPORTANT:
        # NO projection, NO correction (faithful to ReAct)
        response.completion = action
        response.reasoning = raw

        return response

    def _extract_action(self, text: str) -> str:
        """
        ReAct-style extraction:
        take last line or Action: ...
        """
        if not text:
            return "look"

        lines = [l.strip() for l in text.split("\n") if l.strip()]

        for line in reversed(lines):
            if line.lower().startswith("action:"):
                return line.split(":", 1)[1].strip()

        # fallback: last line
        return lines[-1]