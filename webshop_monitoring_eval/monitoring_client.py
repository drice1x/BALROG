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
    monitor: dict[str, Any] | None
    monitor_latest: dict[str, Any] | None
    p_hack: float | None
    p_hack_trajectory: list[float]
    prompt_monitor_prob_so_far: float | None
    prompt_monitor_prob_trajectory: list[float]
    raw_response: dict[str, Any]


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def list_of_floats(values) -> list[float]:
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

    def complete(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, do_sample: bool = True) -> MonitoringResponse:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if do_sample else 0.0,
            "logprobs": True,
            "top_logprobs": 5,
            # OpenAI SDK passes this through via extra_body; over raw HTTP we must send it directly.
            "return_entropy": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.debug:
            debug_payload = {
                "model": payload["model"],
                "max_tokens": payload["max_tokens"],
                "temperature": payload["temperature"],
                "logprobs": payload["logprobs"],
                "top_logprobs": payload["top_logprobs"],
                "return_entropy": payload["return_entropy"],
                "n_messages": len(messages),
            }
            print("[WEBSHOP_MONITOR_DEBUG] request", json.dumps(debug_payload, ensure_ascii=False))
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = response.json()
        if self.debug:
            choice0 = raw.get("choices", [{}])[0]
            dbg = {
                "has_logprobs": "logprobs" in choice0,
                "has_monitor_outputs": "monitor_outputs" in choice0 or "monitor_outputs" in (choice0.get("message") or {}),
                "token_entropy_len": len(self._extract_token_entropy(raw, choice0, choice0.get("message", {}) or {})),
            }
            print("[WEBSHOP_MONITOR_DEBUG] response", json.dumps(dbg, ensure_ascii=False))
        choice = raw["choices"][0]
        message = choice.get("message", {}) or {}
        text = message.get("content", "") or ""

        usage = raw.get("usage", {}) or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)

        token_entropy = self._extract_token_entropy(raw, choice, message)
        entropy_mean = (
            safe_float(choice.get("entropy_mean"))
            or safe_float(message.get("entropy_mean"))
            or (sum(token_entropy) / len(token_entropy) if token_entropy else None)
        )
        entropy_max = (
            safe_float(choice.get("entropy_max"))
            or safe_float(message.get("entropy_max"))
            or (max(token_entropy) if token_entropy else None)
        )

        monitor = self._extract_monitor_blob(raw, choice, message)
        monitor_latest = self._extract_monitor_latest(monitor, raw, choice, message)
        monitor_dict = monitor if isinstance(monitor, dict) else {}
        latest_dict = monitor_latest if isinstance(monitor_latest, dict) else {}
        p_hack = (
            safe_float(monitor_dict.get("p_hack"))
            or safe_float(latest_dict.get("token_score"))
            or safe_float(latest_dict.get("prompt_monitor_prob_so_far"))
            or safe_float(monitor_dict.get("monitor_prob"))
            or safe_float(monitor_dict.get("monitor_latest"))
            or safe_float(choice.get("p_hack"))
            or safe_float(message.get("p_hack"))
            or safe_float(choice.get("prompt_monitor_prob_so_far"))
            or safe_float(message.get("prompt_monitor_prob_so_far"))
        )
        p_hack_trajectory = (
            self._extract_monitor_history(monitor, "token_score")
            or list_of_floats(monitor_dict.get("token_score_trajectory"))
            or list_of_floats(monitor_dict.get("p_hack_trajectory"))
            or list_of_floats(choice.get("token_score_trajectory"))
            or list_of_floats(message.get("token_score_trajectory"))
            or list_of_floats(choice.get("p_hack_trajectory"))
            or list_of_floats(message.get("p_hack_trajectory"))
        )
        prompt_monitor_prob_so_far = (
            safe_float(monitor_dict.get("prompt_monitor_prob_so_far"))
            or safe_float(latest_dict.get("prompt_monitor_prob_so_far"))
            or safe_float(choice.get("prompt_monitor_prob_so_far"))
            or safe_float(message.get("prompt_monitor_prob_so_far"))
        )
        prompt_monitor_prob_trajectory = (
            self._extract_monitor_history(monitor, "prompt_monitor_prob_so_far")
            or list_of_floats(monitor_dict.get("prompt_monitor_prob_trajectory"))
            or list_of_floats(choice.get("prompt_monitor_prob_trajectory"))
            or list_of_floats(message.get("prompt_monitor_prob_trajectory"))
        )

        return MonitoringResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_entropy=token_entropy,
            entropy_mean=entropy_mean,
            entropy_max=entropy_max,
            monitor=monitor if isinstance(monitor, dict) else None,
            monitor_latest=monitor_latest if isinstance(monitor_latest, dict) else None,
            p_hack=p_hack,
            p_hack_trajectory=p_hack_trajectory,
            prompt_monitor_prob_so_far=prompt_monitor_prob_so_far,
            prompt_monitor_prob_trajectory=prompt_monitor_prob_trajectory,
            raw_response=raw,
        )

    def _extract_token_entropy(self, raw: dict[str, Any], choice: dict[str, Any], message: dict[str, Any]) -> list[float]:
        for container in (choice, message, raw):
            direct = list_of_floats(container.get("token_entropy"))
            if direct:
                return direct
            logprobs = container.get("logprobs", {}) or {}
            direct = list_of_floats(logprobs.get("token_entropy"))
            if direct:
                return direct
            content = logprobs.get("content")
            if isinstance(content, list):
                derived = []
                for entry in content:
                    if isinstance(entry, dict):
                        value = safe_float(entry.get("entropy"))
                        if value is not None:
                            derived.append(value)
                if derived:
                    return derived
        return []

    def _extract_monitor_blob(self, raw: dict[str, Any], choice: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
        for container in (message, choice, raw):
            for key in ("monitor_outputs", "monitor", "monitor_latest"):
                blob = container.get(key)
                if isinstance(blob, dict):
                    return blob
        return {}

    @staticmethod
    def _extract_monitor_latest(monitor: Any, raw: dict[str, Any], choice: dict[str, Any], message: dict[str, Any]) -> dict[str, Any] | None:
        if isinstance(monitor, dict):
            latest = monitor.get("latest")
            if isinstance(latest, dict):
                return latest
        for container in (message, choice, raw):
            latest = container.get("monitor_latest")
            if isinstance(latest, dict):
                return latest
        return None

    @staticmethod
    def _extract_monitor_history(monitor: Any, key: str) -> list[float]:
        if not isinstance(monitor, dict):
            return []
        history = monitor.get("history")
        if not isinstance(history, list):
            return []
        values = []
        for item in history:
            if isinstance(item, dict):
                value = safe_float(item.get(key))
                if value is not None:
                    values.append(value)
        return values
