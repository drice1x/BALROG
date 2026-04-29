from __future__ import annotations

from difflib import get_close_matches
from typing import Any

import gymnasium as gym
from gymnasium import spaces

import gym as old_gym
from web_agent_site.envs import WebAgentTextEnv  # noqa: F401


class WebShopTextEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task: str, config):
        super().__init__()
        self.task = task
        self.config = config
        self.max_steps = getattr(config.eval, "max_steps_per_episode", 30) or 30
        self.failed_candidates = []

        num_products = 1000
        if hasattr(config.envs, "webshop_kwargs"):
            num_products = int(getattr(config.envs.webshop_kwargs, "num_products", num_products))

        try:
            from web_agent_site.envs import WebAgentTextEnv  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "WebShop is not installed. Install the WebShop repo/environment "
                "before using envs.names=webshop."
            ) from e

        self.env = old_gym.make(
            "WebAgentTextEnv-v0",
            observation_mode="text",
            num_products=num_products,
        )

        self._steps = 0
        self._last_obs = None
        self._last_info = None
        self._clickables = []
        self._has_search_bar = False
        self._instruction = task

        self.observation_space = spaces.Dict({
            "text": spaces.Dict({
                "long_term_context": spaces.Text(max_length=200000),
            }),
            "image": spaces.Text(max_length=10),
            "mission": spaces.Text(max_length=5000),
            "webshop_clickables": spaces.Sequence(spaces.Text(max_length=256)),
            "webshop_has_search_bar": spaces.Discrete(2),
        })
        self.action_space = spaces.Text(max_length=512)

    @property
    def language_action_space(self):
        acts = [f"click[{x}]" for x in self._clickables]
        if self._has_search_bar:
            acts.append("search[<keywords>]")
        return acts

    @property
    def default_action(self):
        if self._clickables:
            return f"click[{self._clickables[0]}]"
        if self._has_search_bar:
            return "search[]"
        return "click[search]"

    def reset(self, seed=None, options=None):
        self._steps = 0
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        self._last_info = info or {}
        self._instruction = self._extract_instruction(obs, info)
        self._sync_valid_actions(info, obs)
        self._last_obs = self._format_obs(obs, info)
        return self._last_obs, info

    def step(self, action: str):
        self._steps += 1
        valid_action = self.check_action_validity(action)

        result = self.env.step(valid_action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = self._steps >= self.max_steps and not terminated
        else:
            obs, reward, terminated, truncated, info = result

        self._last_info = info or {}
        self._sync_valid_actions(info, obs)
        self._last_obs = self._format_obs(obs, info)

        return self._last_obs, float(reward), bool(terminated), bool(truncated), info

    def _sync_valid_actions(self, info: dict[str, Any], obs: Any):
        clickables = []

        if isinstance(info, dict):
            if "valid" in info and isinstance(info["valid"], (list, tuple)):
                clickables = [str(x).strip() for x in info["valid"] if str(x).strip()]
            elif "clickables" in info and isinstance(info["clickables"], (list, tuple)):
                clickables = [str(x).strip() for x in info["clickables"] if str(x).strip()]
            elif "available_actions" in info and isinstance(info["available_actions"], dict):
                clickables = [str(x).strip() for x in info["available_actions"].get("clickables", []) if str(x).strip()]

            if "has_search_bar" in info:
                self._has_search_bar = bool(info["has_search_bar"])
            elif "available_actions" in info and isinstance(info["available_actions"], dict):
                self._has_search_bar = bool(info["available_actions"].get("has_search_bar", False))
            else:
                self._has_search_bar = "search" in [c.lower() for c in clickables]
        else:
            self._has_search_bar = False

        self._clickables = clickables

    def check_action_validity(self, action: str) -> str:
        action = (action or "").strip()

        if self._has_search_bar:
            m = re_match_search(action)
            if m is not None:
                return f"search[{m}]"

        click_actions = [f"click[{c}]" for c in self._clickables]
        if action in click_actions:
            return action

        m = re_match_click(action)
        if m is not None:
            target = m
            lower_clicks = [a.lower() for a in click_actions]
            exact = f"click[{target}]".lower()
            if exact in lower_clicks:
                return click_actions[lower_clicks.index(exact)]

            match = get_close_matches(exact, lower_clicks, n=1, cutoff=0.6)
            if match:
                self.failed_candidates.append(action)
                return click_actions[lower_clicks.index(match[0])]

        self.failed_candidates.append(action)
        return self.default_action

    def get_instruction_prompt(self, instructions=None):
        return (
            "You are acting in WebShop, a text shopping environment. "
            "At each step choose one valid action. "
            "Use search[keywords] only when a search bar is available. "
            "Use click[value] only for available clickable values."
        )

    def get_stats(self):
        reward = 0.0
        if isinstance(self._last_info, dict):
            try:
                reward = float(self._last_info.get("reward", 0.0))
            except (TypeError, ValueError):
                reward = 0.0

        return {
            "progression": reward,
            "success": reward,
            "webshop_reward": reward,
        }

    def _extract_instruction(self, obs: Any, info: dict[str, Any]) -> str:
        if isinstance(info, dict) and "instruction" in info:
            return str(info["instruction"])
        if isinstance(obs, str):
            return obs
        return str(self.task)

    def _format_obs(self, obs: Any, info: dict[str, Any]):
        obs_text = obs if isinstance(obs, str) else str(obs)
        click_block = "\n".join(f"- click[{c}]" for c in self._clickables[:80]) if self._clickables else "- none"
        search_line = "search[keywords] is available." if self._has_search_bar else "search is unavailable."

        return {
            "text": {
                "long_term_context": (
                    f"{obs_text}\n\n"
                    f"Instruction:\n{self._instruction}\n\n"
                    f"Available click actions:\n{click_block}\n\n"
                    f"{search_line}"
                )
            },
            "image": None,
            "mission": self._instruction,
            "webshop_clickables": list(self._clickables),
            "webshop_has_search_bar": self._has_search_bar,
        }

    def close(self):
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()


def re_match_search(action: str) -> str | None:
    import re
    m = re.fullmatch(r"search\[(.*)\]", action, flags=re.DOTALL)
    if m is None:
        return None
    return m.group(1).strip()


def re_match_click(action: str) -> str | None:
    import re
    m = re.fullmatch(r"click\[(.*)\]", action, flags=re.DOTALL)
    if m is None:
        return None
    return m.group(1).strip()