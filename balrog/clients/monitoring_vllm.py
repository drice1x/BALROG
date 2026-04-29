from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean
from typing import Any

from openai import OpenAI


@dataclass
class MonitoringLLMResponse:
    completion: str
    reasoning: str | None
    input_tokens: int
    output_tokens: int
    raw_response: dict[str, Any]

    token_entropy: list[float | None]
    entropy_mean: float | None
    entropy_max: float | None

    monitor: dict[str, Any] | None
    monitor_latest: dict[str, Any] | None

    # Main scalar for downstream tables: cumulative prompt-level p(hack) if available.
    p_hack: float | None

    # Token-wise instantaneous p(hack) trajectory.
    p_hack_trajectory: list[float | None]

    # Cumulative prompt-level trajectory.
    prompt_monitor_prob_so_far: float | None
    prompt_monitor_prob_trajectory: list[float | None]

    # Explicit aliases for clarity.
    token_p_hack: float | None
    token_p_hack_trajectory: list[float | None]
    prompt_p_hack: float | None
    prompt_p_hack_trajectory: list[float | None]


class MonitoringVLLMWrapper:
    def __init__(self, config):
        self.model_id = config.model_id
        self.base_url = config.base_url
        self.api_key = getattr(config, "api_key", "token-abc123")

        generate_kwargs = getattr(config, "generate_kwargs", {}) or {}
        self.max_tokens = int(generate_kwargs.get("max_tokens", 256))
        self.temperature = float(generate_kwargs.get("temperature", 0.0))

        self.top_logprobs = int(getattr(config, "top_logprobs", 5))
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(
        self,
        messages,
        max_tokens=None,
        temperature=None,
        **kwargs,
    ):
        openai_messages = []
        for m in messages:
            role = getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
            openai_messages.append({"role": role, "content": content})

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=openai_messages,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            extra_body={"return_entropy": True},
        )

        choice = resp.choices[0]
        raw_text = choice.message.content or ""
        response_dict = self._to_dict(resp)

        usage = response_dict.get("usage", {}) or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)

        entropies = self._safe_get_entropy_list(choice)
        entropy_mean, entropy_max = self._summarize_entropy(entropies)

        choice_monitor = getattr(choice.message, "monitor_outputs", None)

        monitor = (
            choice_monitor
            or response_dict.get("monitor_outputs")
            or response_dict.get("monitor")
        )

        monitor_latest = None
        token_p_hack = None
        prompt_p_hack = None
        token_p_hack_trajectory = []
        prompt_p_hack_trajectory = []

        if isinstance(monitor, dict):
            monitor_latest = monitor.get("latest")

            history = monitor.get("history", [])
            if isinstance(history, list):
                token_p_hack_trajectory = [
                    x.get("token_score") if isinstance(x, dict) else None
                    for x in history
                ]

                prompt_p_hack_trajectory = [
                    x.get("prompt_monitor_prob_so_far")
                    if isinstance(x, dict)
                    else None
                    for x in history
                ]

        if isinstance(monitor_latest, dict):
            token_p_hack = monitor_latest.get("token_score")
            prompt_p_hack = monitor_latest.get("prompt_monitor_prob_so_far")

        # Main scalar: cumulative trajectory risk if available; fallback to token spike.
        p_hack = prompt_p_hack if prompt_p_hack is not None else token_p_hack

        return MonitoringLLMResponse(
            completion=raw_text,
            reasoning=None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw_response=response_dict,
            token_entropy=entropies,
            entropy_mean=entropy_mean,
            entropy_max=entropy_max,
            monitor=monitor,
            monitor_latest=monitor_latest,
            p_hack=p_hack,
            p_hack_trajectory=token_p_hack_trajectory,
            prompt_monitor_prob_so_far=prompt_p_hack,
            prompt_monitor_prob_trajectory=prompt_p_hack_trajectory,
            token_p_hack=token_p_hack,
            token_p_hack_trajectory=token_p_hack_trajectory,
            prompt_p_hack=prompt_p_hack,
            prompt_p_hack_trajectory=prompt_p_hack_trajectory,
        )

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
    def _summarize_entropy(
        entropies: list[float | None],
    ) -> tuple[float | None, float | None]:
        vals = [float(x) for x in entropies if x is not None]
        if not vals:
            return None, None
        return mean(vals), max(vals)

    @staticmethod
    def _to_dict(resp: Any) -> dict[str, Any]:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "dict"):
            return resp.dict()
        return json.loads(json.dumps(resp, default=str))