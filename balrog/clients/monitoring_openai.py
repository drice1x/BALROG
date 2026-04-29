from __future__ import annotations

import json
from statistics import mean
from typing import Any

from openai import OpenAI


class MonitoringResponse:
    def __init__(
        self,
        completion: str,
        reasoning: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        raw_response: dict[str, Any] | None = None,
        token_entropy: list[float | None] | None = None,
        entropy_mean: float | None = None,
        entropy_max: float | None = None,
        monitor: dict[str, Any] | None = None,
        monitor_latest: dict[str, Any] | None = None,
        p_hack: float | None = None,
        p_hack_trajectory: list[float | None] | None = None,
        prompt_monitor_prob_so_far: float | None = None,
        prompt_monitor_prob_trajectory: list[float | None] | None = None,
    ):
        self.completion = completion
        self.reasoning = reasoning
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.raw_response = raw_response or {}
        self.token_entropy = token_entropy or []
        self.entropy_mean = entropy_mean
        self.entropy_max = entropy_max
        self.monitor = monitor
        self.monitor_latest = monitor_latest
        self.p_hack = p_hack
        self.p_hack_trajectory = p_hack_trajectory or []
        self.prompt_monitor_prob_so_far = prompt_monitor_prob_so_far
        self.prompt_monitor_prob_trajectory = prompt_monitor_prob_trajectory or []


class MonitoringOpenAIWrapper:
    def __init__(self, client_config):
        self.client_config = client_config
        self.model_id = client_config.model_id
        self.base_url = client_config.base_url
        self.api_key = getattr(client_config, "api_key", "token-abc123")

        generate_kwargs = getattr(client_config, "generate_kwargs", {})
        self.max_new_tokens = int(generate_kwargs.get("max_tokens", 256))
        self.temperature = float(generate_kwargs.get("temperature", 0.0))
        self.top_logprobs = int(getattr(client_config, "top_logprobs", 5))

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
    def generate(self, messages, **kwargs):
        openai_messages = []
        for m in messages:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
            openai_messages.append({"role": role, "content": content})

        max_tokens = int(kwargs.get("max_tokens", self.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            extra_body={"return_entropy": True},
        )

        choice = resp.choices[0]
        raw_text = choice.message.content or ""
        raw_dict = self._to_dict(resp)

        token_entropy = self._safe_get_entropy_list(choice)
        entropy_mean, entropy_max = self._summarize_entropy(token_entropy)

        usage = raw_dict.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        monitor = raw_dict.get("monitor")
        monitor_latest = None
        p_hack = None
        p_hack_trajectory = []
        prompt_monitor_prob_so_far = None
        prompt_monitor_prob_trajectory = []

        if isinstance(monitor, dict):
            monitor_latest = monitor.get("latest", monitor)
            history = monitor.get("history", [])
            if isinstance(history, list):
                p_hack_trajectory = [
                    h.get("token_score") if isinstance(h, dict) else None
                    for h in history
                ]
                prompt_monitor_prob_trajectory = [
                    h.get("prompt_monitor_prob_so_far") if isinstance(h, dict) else None
                    for h in history
                ]

        if isinstance(monitor_latest, dict):
            p_hack = monitor_latest.get("token_score")
            prompt_monitor_prob_so_far = monitor_latest.get("prompt_monitor_prob_so_far")

        return MonitoringResponse(
            completion=raw_text,
            reasoning="",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=raw_dict,
            token_entropy=token_entropy,
            entropy_mean=entropy_mean,
            entropy_max=entropy_max,
            monitor=monitor,
            monitor_latest=monitor_latest,
            p_hack=p_hack,
            p_hack_trajectory=p_hack_trajectory,
            prompt_monitor_prob_so_far=prompt_monitor_prob_so_far,
            prompt_monitor_prob_trajectory=prompt_monitor_prob_trajectory,
        )
    def generateOLD(self, messages, **kwargs):
        openai_messages = []
        for m in messages:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
            openai_messages.append({"role": role, "content": content})

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=openai_messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            extra_body={"return_entropy": True},
        )

        choice = resp.choices[0]
        raw_text = choice.message.content or ""
        raw_dict = self._to_dict(resp)

        token_entropy = self._safe_get_entropy_list(choice)
        entropy_mean, entropy_max = self._summarize_entropy(token_entropy)

        usage = raw_dict.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        monitor = raw_dict.get("monitor")
        monitor_latest = None
        p_hack = None
        p_hack_trajectory = []
        prompt_monitor_prob_so_far = None
        prompt_monitor_prob_trajectory = []

        if isinstance(monitor, dict):
            monitor_latest = monitor.get("latest", monitor)
            history = monitor.get("history", [])
            if isinstance(history, list):
                p_hack_trajectory = [
                    h.get("token_score") if isinstance(h, dict) else None
                    for h in history
                ]
                prompt_monitor_prob_trajectory = [
                    h.get("prompt_monitor_prob_so_far") if isinstance(h, dict) else None
                    for h in history
                ]

        if isinstance(monitor_latest, dict):
            p_hack = monitor_latest.get("token_score")
            prompt_monitor_prob_so_far = monitor_latest.get("prompt_monitor_prob_so_far")

        return MonitoringResponse(
            completion=raw_text,
            reasoning="",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=raw_dict,
            token_entropy=token_entropy,
            entropy_mean=entropy_mean,
            entropy_max=entropy_max,
            monitor=monitor,
            monitor_latest=monitor_latest,
            p_hack=p_hack,
            p_hack_trajectory=p_hack_trajectory,
            prompt_monitor_prob_so_far=prompt_monitor_prob_so_far,
            prompt_monitor_prob_trajectory=prompt_monitor_prob_trajectory,
        )

    @staticmethod
    def _to_dict(resp: Any) -> dict[str, Any]:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "dict"):
            return resp.dict()
        return json.loads(json.dumps(resp, default=str))

    @staticmethod
    def _safe_get_entropy_list(choice: Any) -> list[float | None]:
        out = []
        logprobs = getattr(choice, "logprobs", None)
        if logprobs is None:
            return out
        content = getattr(logprobs, "content", None)
        if content is None:
            return out
        for item in content:
            out.append(getattr(item, "entropy", None))
        return out

    @staticmethod
    def _summarize_entropy(entropies: list[float | None]) -> tuple[float | None, float | None]:
        vals = [float(x) for x in entropies if x is not None]
        if not vals:
            return None, None
        return mean(vals), max(vals)