from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
BALROG_ROOT = THIS_DIR.parent
if str(BALROG_ROOT) not in sys.path:
    sys.path.append(str(BALROG_ROOT))

from react_webshop_agent import (
    ASIN_RE,
    build_broad_search_query,
    build_react_webshop_prompt,
    build_task_query,
    choose_best_asin_action,
    selected_options,
)
from hf_alfworld_steering.risk_steering import GatedActivationSteerer


@dataclass
class HFWebShopStep:
    action: str
    raw_action: str
    thought_text: str
    steering: dict


class HFSteeringWebShopAgent:
    def __init__(
        self,
        base_model: str,
        lora_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        directions: dict[int, torch.Tensor] | None = None,
        direction_path: str | None = None,
        alpha: float = 0.0,
        steering_mode: str = "always",
        steering_tau: float = 0.0,
        steering_token_scope: str = "last",
        max_tokens: int = 100,
        temperature: float = 0.0,
    ):
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, lora_path)
        self.model.eval()

        self.device = device
        self.alpha = float(alpha)
        self.steering_mode = steering_mode
        self.steering_tau = float(steering_tau)
        self.steering_token_scope = steering_token_scope
        self.directions = self._load_directions(directions=directions, direction_path=direction_path)

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history: list[dict[str, str]] = []
        self.last_steering_summary = self._default_steering_summary()

    def reset(self) -> None:
        self.history = []
        self.last_steering_summary = self._default_steering_summary()

    def act(self, observation: dict) -> HFWebShopStep:
        prompt = self._build_prompt(observation)
        with self._steering_context() as steerer:
            raw_action = self._generate_one_line(prompt)
        if steerer is not None:
            self.last_steering_summary = steerer.summary()
        else:
            self.last_steering_summary = self._default_steering_summary()

        action = self._extract_action(raw_action)
        action = self._normalize_action(action, observation)
        if not action:
            best_asin = choose_best_asin_action(observation)
            if best_asin:
                action = best_asin
            elif "click[Buy Now]" in (observation.get("available_actions", []) or []) and selected_options(observation):
                action = "click[Buy Now]"
            else:
                action = "search[" + build_broad_search_query(observation.get("mission", "")) + "]"
        action = self._finalize_action(action, observation)

        self.history.append(
            {
                "observation": observation["text"]["long_term_context"],
                "action": action,
            }
        )
        return HFWebShopStep(
            action=action,
            raw_action=raw_action,
            thought_text="",
            steering=dict(self.last_steering_summary),
        )

    def _build_prompt(self, observation: dict) -> str:
        return build_react_webshop_prompt(observation, self.history)

    def _format_history(self) -> str:
        if not self.history:
            return "[Search]\n\n"
        lines = []
        for item in self.history[-4:]:
            lines.append(f"Action: {item['action']}")
            lines.append(f"Observation:\n{item['observation']}")
        return "\n".join(lines) + "\n\n"

    @torch.no_grad()
    def _generate_one_line(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # Some decoder-only model families surface token_type_ids from the tokenizer
        # but reject them during generation.
        inputs.pop("token_type_ids", None)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return text.split("\n")[0].strip()

    @staticmethod
    def _extract_action(text: str) -> str:
        matches = re.findall(r"Action:\s*(search\[[^\]]*\]|click\[[^\]]*\]|think\[[^\]]*\])", text, flags=re.I)
        if matches:
            return matches[-1].strip()
        matches = re.findall(r"(search\[[^\]]*\]|click\[[^\]]*\]|think\[[^\]]*\])", text, flags=re.I)
        for candidate in reversed(matches):
            lowered = candidate.strip().lower()
            if lowered not in {"search[query]", "search[keywords]"}:
                return candidate.strip()
        return ""

    @staticmethod
    def _normalize_action(action: str, observation: dict) -> str:
        if not action:
            return ""
        instruction = observation.get("mission", "") or ""
        fallback_query = build_task_query(instruction)
        broad_query = build_broad_search_query(instruction)
        best_asin = choose_best_asin_action(observation)
        selected = selected_options(observation)
        available_actions = observation.get("available_actions", []) or []
        lowered = action.lower()
        if lowered.startswith("think["):
            return action
        if lowered in {"search[query]", "search[keywords]"}:
            return f"search[{fallback_query}]"
        if lowered.startswith("search[instruction:"):
            return f"search[{fallback_query}]"
        if lowered == "search[3 ounce bright citrus deodorant sensitive skin]":
            return f"search[{fallback_query}]"
        if lowered.startswith("search[") and best_asin:
            return best_asin
        if lowered.startswith("click["):
            button = action[6:-1].strip().lower()
            if button in selected and "click[Buy Now]" in available_actions:
                return "click[Buy Now]"
        if lowered.startswith("search["):
            query = action[7:-1].strip()
            query_tokens = query.split()
            too_specific = (
                len(query_tokens) > 5
                or re.search(r"(?i)\b(size|color|colour|price|dollars|lower|than|inch|inches)\b", query)
                or query.lower() == fallback_query.lower()
            )
            if too_specific:
                return f"search[{broad_query}]"
        return action

    @staticmethod
    def _finalize_action(action: str, observation: dict) -> str:
        if not action:
            return action
        available_actions = observation.get("available_actions", []) or []
        selected = selected_options(observation)
        if "click[Buy Now]" not in available_actions or not selected:
            return action
        lowered = action.lower()
        if lowered.startswith("click["):
            button = action[6:-1].strip().lower()
            if button in selected:
                return "click[Buy Now]"
        return action

    def _load_directions(self, directions, direction_path):
        if directions:
            return directions
        if not direction_path:
            return {}
        obj = torch.load(direction_path, map_location="cpu")
        if isinstance(obj, dict) and "directions" in obj:
            obj = obj["directions"]
        if not isinstance(obj, dict):
            raise ValueError(f"Direction file must contain a dict, got {type(obj)}")
        return {int(layer): tensor for layer, tensor in obj.items()}

    def _default_steering_summary(self):
        return {
            "steering_mode": self.steering_mode,
            "steering_alpha": self.alpha,
            "steering_tau": self.steering_tau,
            "steering_token_scope": self.steering_token_scope,
            "steering_num_seen": 0,
            "steering_num_steered": 0,
            "steering_rate": 0.0,
            "steering_risk_mean": 0.0,
            "steering_risk_max": 0.0,
        }

    def _steering_context(self):
        if not self.directions or self.alpha == 0.0:
            return _NullSteeringContext()
        return GatedActivationSteerer(
            self.model,
            self.directions,
            self.alpha,
            mode=self.steering_mode,
            tau=self.steering_tau,
            token_scope=self.steering_token_scope,
        )


class _NullSteeringContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
