from __future__ import annotations

import re
import torch
from difflib import get_close_matches
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from steering import ActivationSteerer


class HFSteeringAgent:
    def __init__(
        self,
        base_model: str,
        lora_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        directions: dict[int, torch.Tensor] | None = None,
        alpha: float = 0.0,
        reasoning_tokens: int = 32,
        action_tokens: int = 12,
        reasoning_temperature: float = 0.7,
        action_temperature: float = 0.0,
        self.steering_mode = steering_mode
        self.steering_tau = steering_tau
        self.last_steering_summary = {}
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
        self.directions = directions or {}
        self.alpha = alpha

        self.reasoning_tokens = reasoning_tokens
        self.action_tokens = action_tokens
        self.reasoning_temperature = reasoning_temperature
        self.action_temperature = action_temperature

        self.history = []

    def reset(self):
        self.history = []

    def act(self, obs: dict):
        valid_actions = obs.get("admissible_commands", [])
        valid_actions = [str(a).strip().lower() for a in valid_actions if str(a).strip()]
        default_action = "look" if "look" in valid_actions else (valid_actions[0] if valid_actions else "look")

        observation = obs["text"]["long_term_context"]
        task = obs.get("mission", "")

        reasoning_prompt = self._build_reasoning_prompt(task, observation, valid_actions)
        reasoning = self._generate(
            reasoning_prompt,
            max_new_tokens=self.reasoning_tokens,
            temperature=self.reasoning_temperature,
            do_sample=self.reasoning_temperature > 0,
        )

        action_prompt = self._build_action_prompt(task, observation, valid_actions, reasoning)
        with ActivationSteerer(self.model, self.directions, self.alpha):
            raw_action = self._generate(
                action_prompt,
                max_new_tokens=self.action_tokens,
                temperature=self.action_temperature,
                do_sample=False,
            )

        action = self._parse_action(raw_action)
        action = self._project_action(action, valid_actions, default_action)

        self.history.append(
            {
                "observation": observation,
                "reasoning": reasoning,
                "raw_action": raw_action,
                "action": action,
            }
        )

        return action, reasoning, raw_action

    def _build_reasoning_prompt(self, task, obs, valid_actions):
        valid = "\n".join(f"- {a}" for a in valid_actions[:60])
        return f"""You are acting in ALFWorld.

Task: {task}

Observation:
{obs}

Valid actions:
{valid}

Think briefly about the best next action.
Reasoning:"""

    def _build_action_prompt(self, task, obs, valid_actions, reasoning):
        valid = "\n".join(f"- {a}" for a in valid_actions[:60])
        return f"""You are acting in ALFWorld.

Task: {task}

Observation:
{obs}

Reasoning:
{reasoning}

Valid actions:
{valid}

You must choose EXACTLY ONE action from the valid action list.
Do NOT modify spelling.
Do NOT invent actions.
Return only the action string.

Action:"""

    @torch.no_grad()
    def _generate(self, prompt, max_new_tokens, temperature, do_sample):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def _parse_action(self, text):
        text = (text or "").strip().lower()
        text = text.split("\n")[0].strip()
        text = re.sub(r"^action\s*:\s*", "", text)
        return text

    def _project_action(self, action, valid_actions, default):
        if action in valid_actions:
            return action
        match = get_close_matches(action, valid_actions, n=1, cutoff=0.75)
        return match[0] if match else default