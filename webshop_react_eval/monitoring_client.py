from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

import requests


@dataclass
class MonitoringResponse:
    text: str
    input_tokens: int
    output_tokens: int
    token_entropy: list[float]
    entropy_mean: float | None
    entropy_max: float | None
    p_hack: float | None
    p_hack_trajectory: list[float]
    prompt_monitor_prob_so_far: float | None
    prompt_monitor_prob_trajectory: list[float]
    raw_response: dict[str, Any]


def safe_float(value):
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def floats(values) -> list[float]:
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        number = safe_float(value)
        if number is not None:
            out.append(number)
    return out


class MonitoringClient:
    def __init__(self, base_url: str, model_id: str, api_key: str = "EMPTY", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        self.debug = os.getenv("WEBSHOP_MONITOR_DEBUG", "0") == "1"

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
    ) -> MonitoringResponse:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 5,
            "return_entropy": True,
        }
        if stop:
            payload["stop"] = stop
        if self.debug:
            print("[WEBSHOP_REACT_DEBUG] request", json.dumps(payload, ensure_ascii=False))
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = response.json()
        choice = raw["choices"][0]
        message = choice.get("message", {}) or {}
        text = message.get("content", "") or ""

        usage = raw.get("usage", {}) or {}
        token_entropy = self._extract_token_entropy(raw, choice, message)
        monitor = self._extract_monitor(raw, choice, message)
        latest = monitor.get("latest", {}) if isinstance(monitor, dict) else {}

        return MonitoringResponse(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            token_entropy=token_entropy,
            entropy_mean=(
                safe_float(choice.get("entropy_mean"))
                or safe_float(message.get("entropy_mean"))
                or (sum(token_entropy) / len(token_entropy) if token_entropy else None)
            ),
            entropy_max=(
                safe_float(choice.get("entropy_max"))
                or safe_float(message.get("entropy_max"))
                or (max(token_entropy) if token_entropy else None)
            ),
            p_hack=(
                safe_float(latest.get("token_score"))
                or safe_float(monitor.get("p_hack") if isinstance(monitor, dict) else None)
                or safe_float(choice.get("p_hack"))
                or safe_float(message.get("p_hack"))
            ),
            p_hack_trajectory=(
                self._extract_monitor_history(monitor, "token_score")
                or floats(choice.get("p_hack_trajectory"))
                or floats(message.get("p_hack_trajectory"))
            ),
            prompt_monitor_prob_so_far=(
                safe_float(latest.get("prompt_monitor_prob_so_far"))
                or safe_float(choice.get("prompt_monitor_prob_so_far"))
                or safe_float(message.get("prompt_monitor_prob_so_far"))
            ),
            prompt_monitor_prob_trajectory=(
                self._extract_monitor_history(monitor, "prompt_monitor_prob_so_far")
                or floats(choice.get("prompt_monitor_prob_trajectory"))
                or floats(message.get("prompt_monitor_prob_trajectory"))
            ),
            raw_response=raw,
        )

    @staticmethod
    def _extract_monitor(raw: dict[str, Any], choice: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
        for container in (message, choice, raw):
            for key in ("monitor_outputs", "monitor", "monitor_latest"):
                blob = container.get(key)
                if isinstance(blob, dict):
                    return blob
        return {}

    @staticmethod
    def _extract_monitor_history(monitor: Any, key: str) -> list[float]:
        if not isinstance(monitor, dict):
            return []
        history = monitor.get("history")
        if not isinstance(history, list):
            return []
        out = []
        for item in history:
            if isinstance(item, dict):
                value = safe_float(item.get(key))
                if value is not None:
                    out.append(value)
        return out

    @staticmethod
    def _extract_token_entropy(raw: dict[str, Any], choice: dict[str, Any], message: dict[str, Any]) -> list[float]:
        for container in (choice, message, raw):
            direct = floats(container.get("token_entropy"))
            if direct:
                return direct
            logprobs = container.get("logprobs", {}) or {}
            direct = floats(logprobs.get("token_entropy"))
            if direct:
                return direct
            content = logprobs.get("content")
            if isinstance(content, list):
                out = []
                for entry in content:
                    if isinstance(entry, dict):
                        value = safe_float(entry.get("entropy"))
                        if value is not None:
                            out.append(value)
                if out:
                    return out
        return []
