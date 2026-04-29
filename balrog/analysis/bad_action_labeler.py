# balrog/analysis/bad_action_labeler.py
from __future__ import annotations

from collections import deque
from typing import Any


BABYAI_VALID_ACTIONS = {
    "left",
    "right",
    "forward",
    "pickup",
    "drop",
    "toggle",
    "done",
}

MINIHACK_VALID_ACTIONS = {
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
    "search",
    "kick",
    "open",
    "eat",
    "wait",
}

TEXTWORLD_VALID_ACTIONS = {
    "goal",
    "look",
    "inventory",
    "go north",
    "go east",
    "go south",
    "go west",
    "take coin",
}
ALFWORLD_SAFE_DEFAULTS = {"look", "inventory", "examine"}

class StepLabeler:
    """
    Environment-grounded step labeler.

    Goal:
    define "bad_action" independently of the monitor.

    First version uses simple, robust heuristics:
    - invalid_action
    - repeat_action_loop
    - no_progress_window

    This is enough for:
    A) construct validity
    B) predictive validity
    later C) intervention utility
    """

    def __init__(
        self,
        env_name: str,
        loop_window: int = 4,
        no_progress_window: int = 6,
    ):
        self.env_name = env_name
        self.loop_window = loop_window
        self.no_progress_window = no_progress_window

        self.action_hist = deque(maxlen=max(loop_window, no_progress_window))
        self.reward_hist = deque(maxlen=no_progress_window)
        self.obs_hist = deque(maxlen=no_progress_window)

    def reset(self):
        self.action_hist.clear()
        self.reward_hist.clear()
        self.obs_hist.clear()

    def label_step(
        self,
        action: str,
        reward: float,
        obs_before: Any,
        obs_after: Any,
        done: bool = False,
        action_was_rewritten: bool = False,
    ) -> dict:
        invalid_action = action_was_rewritten if self.env_name == "alfworld" else self._invalid_action(action)
        repeat_action_loop = self._repeat_action_loop(action)
        no_progress_window = self._no_progress_window(reward, obs_after, done=done)

        obs_before_fp = self._obs_fingerprint(obs_before)
        obs_after_fp = self._obs_fingerprint(obs_after)

        self.action_hist.append(action)
        self.reward_hist.append(reward)
        self.obs_hist.append(obs_after_fp)

        # primary label
        bad_action = no_progress_window

        return {
            "invalid_action": invalid_action,
            "repeat_action_loop": repeat_action_loop,
            "no_progress_window": no_progress_window,
            "bad_action": bad_action,
            "observation_changed": obs_before_fp != obs_after_fp,
            "recent_action_history": list(self.action_hist),
        }

    def _invalid_action(self, action: str) -> bool:
        action = (action or "").strip().lower()

        if self.env_name == "babyai":
            return action not in BABYAI_VALID_ACTIONS

        if self.env_name == "minihack":
            return action not in MINIHACK_VALID_ACTIONS

        if self.env_name == "textworld":
            return action not in TEXTWORLD_VALID_ACTIONS

        if self.env_name == "alfworld":
            # invalidity should be logged from evaluator if action was rewritten
            return False

        return False

    def _repeat_action_loop(self, action: str) -> bool:
        tmp = list(self.action_hist) + [action]

        if len(tmp) < self.loop_window:
            return False

        # ABAB loop pattern, e.g. left-right-left-right
        if len(tmp) >= 4:
            a, b, c, d = tmp[-4:]
            if a == c and b == d and a != b:
                return True

        # AAAA repeated same action
        if len(tmp) >= 4 and len(set(tmp[-4:])) == 1:
            return True

        return False

    def _no_progress_window(self, reward: float, obs_after: Any, done: bool = False) -> bool:
        if done:
            return False

        tmp_rewards = list(self.reward_hist) + [reward]
        tmp_obs = list(self.obs_hist) + [self._obs_fingerprint(obs_after)]

        if len(tmp_rewards) < self.no_progress_window:
            return False

        recent_rewards = tmp_rewards[-self.no_progress_window:]
        recent_obs = tmp_obs[-self.no_progress_window:]

        # no positive reward in the whole window
        no_positive_reward = all(r <= 0 for r in recent_rewards)

        # almost same observation over the full window
        low_observation_change = len(set(recent_obs)) <= 2

        return no_positive_reward and low_observation_change

    def _obs_fingerprint(self, obs: Any) -> str:
        """
        Convert environment observation into a compact comparable string.

        We only need something stable enough to detect repeated states.
        """
        if obs is None:
            return ""

        if isinstance(obs, str):
            return obs.strip()

        if isinstance(obs, dict):
            chunks = []

            # BALROG text obs often lives here
            text_part = obs.get("text")
            if isinstance(text_part, dict):
                for v in text_part.values():
                    if isinstance(v, str):
                        chunks.append(v.strip())

            # mission field for BabyAI
            mission = obs.get("mission")
            if isinstance(mission, str):
                chunks.append(mission.strip())

            # any other top-level strings
            for k, v in obs.items():
                if isinstance(v, str) and k != "mission":
                    chunks.append(v.strip())

            return " | ".join(chunks)

        return str(obs)