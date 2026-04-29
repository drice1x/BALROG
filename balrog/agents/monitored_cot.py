import copy
import re

from balrog.agents.base import BaseAgent
from balrog.client import LLMClientWrapper


class MonitoredCoTAgent(BaseAgent):
    """BALROG CoT agent with monitoring/entropy trace capture."""

    def __init__(self, client_factory: LLMClientWrapper, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        self.remember_cot = config.agent.remember_cot
        self.last_trace = None

    def reset(self):
        super().reset()
        self.last_trace = None

    def act(self, obs, prev_action=None):
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)

        messages = self.prompt_builder.get_prompt()

        cot_instructions = """
First think about what's the best course of action step by step.
Finally, provide a single output action at the end of the message in the form of: ACTION: <action>
        """.strip()

        messages[-1].content += "\n\n" + cot_instructions

        cot_reasoning = self.client.generate(messages)

        # Save monitor/entropy trace BEFORE parsing final action
        self.last_trace = {
            "raw_completion": getattr(cot_reasoning, "completion", None),
            "token_entropy": getattr(cot_reasoning, "token_entropy", []),
            "entropy_mean": getattr(cot_reasoning, "entropy_mean", None),
            "entropy_max": getattr(cot_reasoning, "entropy_max", None),
            "monitor": getattr(cot_reasoning, "monitor", None),
            "monitor_latest": getattr(cot_reasoning, "monitor_latest", None),
            "p_hack": getattr(cot_reasoning, "p_hack", None),
            "p_hack_trajectory": getattr(cot_reasoning, "p_hack_trajectory", []),
            "prompt_monitor_prob_so_far": getattr(cot_reasoning, "prompt_monitor_prob_so_far", None),
            "prompt_monitor_prob_trajectory": getattr(cot_reasoning, "prompt_monitor_prob_trajectory", []),
            "raw_response": getattr(cot_reasoning, "raw_response", None),
        }

        final_answer = self._extract_final_answer(cot_reasoning)
        return final_answer

    def _extract_final_answer(self, reasoning):
        def filter_letters(input_string):
            return re.sub(r"[^a-zA-Z\s:]", "", input_string)

        answer = copy.deepcopy(reasoning)

        if self.remember_cot:
            self.prompt_builder.update_reasoning(reasoning.completion)

        full_text = answer.completion or ""
        action = filter_letters(full_text).split("ACTION:")[-1].strip()

        # MonitoringResponse is a normal class, not a namedtuple
        answer.reasoning = full_text
        answer.completion = action
        return answer

    def pop_last_trace(self):
        trace = self.last_trace
        self.last_trace = None
        return trace