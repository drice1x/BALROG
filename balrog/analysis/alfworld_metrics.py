from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, asdict
from typing import Any


TASK_RELEVANT_VERBS = (
    "go to ",
    "open ",
    "close ",
    "take ",
    "move ",
    "put ",
    "examine ",
    "use ",
    "heat ",
    "cool ",
    "clean ",
    "slice ",
)


@dataclass
class AlfworldStepLabels:
    repeated_action: bool
    repeated_state_action: bool
    examine_loop: bool
    object_churn: bool
    no_progress_window: bool
    ineffective_action: bool
    reward_irrelevant_action: bool
    productive_action: bool
    unique_action_count_so_far: int
    action_repeat_count_so_far: int
    same_state_repeat_count_so_far: int


@dataclass
class AlfworldEpisodeMetrics:
    repetition_rate: float
    state_action_loop_rate: float
    examine_loop_rate: float
    object_churn_rate: float
    no_progress_window_rate: float
    ineffective_action_rate: float
    reward_irrelevant_action_rate: float
    productive_action_rate: float
    dominant_action_fraction: float
    unique_action_ratio: float
    first_productive_step: int | None
    avg_reward: float
    total_steps: int
    success: float
    progression: float


class AlfworldMetricsTracker:
    """
    Step-level and episode-level metrics for ALFWorld.

    Main metric families:
    - loop metrics
    - progress metrics
    - misalignment-style / waste metrics
    """

    def __init__(
        self,
        no_progress_window: int = 5,
        repeat_window: int = 3,
        state_repeat_window: int = 3,
    ):
        self.no_progress_window = no_progress_window
        self.repeat_window = repeat_window
        self.state_repeat_window = state_repeat_window

        self.actions: list[str] = []
        self.rewards: list[float] = []
        self.obs_fingerprints: list[str] = []
        self.state_action_pairs: list[tuple[str, str]] = []

        self.action_counter: Counter[str] = Counter()
        self.object_counter: Counter[str] = Counter()

        self._recent_actions = deque(maxlen=repeat_window)
        self._recent_states = deque(maxlen=state_repeat_window)
        self._recent_state_actions = deque(maxlen=state_repeat_window)
        self._recent_rewards = deque(maxlen=no_progress_window)

        self.step_labels: list[AlfworldStepLabels] = []
        self.first_productive_step: int | None = None

    def update(
        self,
        obs_before: Any,
        obs_after: Any,
        action: str,
        reward: float,
        success: bool = False,
        progression: float | None = None,
    ) -> AlfworldStepLabels:
        action = (action or "").strip().lower()
        obs_before_fp = self._obs_fingerprint(obs_before)
        obs_after_fp = self._obs_fingerprint(obs_after)
        state_action = (obs_before_fp, action)

        repeated_action = action in self._recent_actions
        repeated_state_action = state_action in self._recent_state_actions
        examine_loop = action.startswith("examine ") and repeated_action
        object_churn = self._is_object_churn(action)
        productive_action = self._is_productive_action(
            obs_before_fp=obs_before_fp,
            obs_after_fp=obs_after_fp,
            reward=reward,
            success=success,
            progression=progression,
            action=action,
        )
        ineffective_action = not productive_action
        reward_irrelevant_action = self._is_reward_irrelevant_action(
            action=action,
            productive_action=productive_action,
            reward=reward,
        )

        self._recent_rewards.append(float(reward))
        no_progress_window = False
        if len(self._recent_rewards) == self.no_progress_window:
            no_progress_window = (
                sum(self._recent_rewards) <= 0.0
                and not success
                and (progression is None or progression <= 0.0)
            )

        self.actions.append(action)
        self.rewards.append(float(reward))
        self.obs_fingerprints.append(obs_after_fp)
        self.state_action_pairs.append(state_action)
        self.action_counter[action] += 1

        obj = self._extract_object_name(action)
        if obj:
            self.object_counter[obj] += 1

        self._recent_actions.append(action)
        self._recent_states.append(obs_before_fp)
        self._recent_state_actions.append(state_action)

        if productive_action and self.first_productive_step is None:
            self.first_productive_step = len(self.actions)

        labels = AlfworldStepLabels(
            repeated_action=repeated_action,
            repeated_state_action=repeated_state_action,
            examine_loop=examine_loop,
            object_churn=object_churn,
            no_progress_window=no_progress_window,
            ineffective_action=ineffective_action,
            reward_irrelevant_action=reward_irrelevant_action,
            productive_action=productive_action,
            unique_action_count_so_far=len(self.action_counter),
            action_repeat_count_so_far=sum(1 for a in self.actions if a == action),
            same_state_repeat_count_so_far=sum(1 for s, a in self.state_action_pairs if s == obs_before_fp and a == action),
        )
        self.step_labels.append(labels)
        return labels

    def finalize(
        self,
        success: float,
        progression: float,
    ) -> AlfworldEpisodeMetrics:
        total_steps = max(len(self.actions), 1)

        repetition_rate = self._mean([x.repeated_action for x in self.step_labels])
        state_action_loop_rate = self._mean([x.repeated_state_action for x in self.step_labels])
        examine_loop_rate = self._mean([x.examine_loop for x in self.step_labels])
        object_churn_rate = self._mean([x.object_churn for x in self.step_labels])
        no_progress_window_rate = self._mean([x.no_progress_window for x in self.step_labels])
        ineffective_action_rate = self._mean([x.ineffective_action for x in self.step_labels])
        reward_irrelevant_action_rate = self._mean([x.reward_irrelevant_action for x in self.step_labels])
        productive_action_rate = self._mean([x.productive_action for x in self.step_labels])

        dominant_action_fraction = 0.0
        if self.action_counter:
            dominant_action_fraction = max(self.action_counter.values()) / total_steps

        unique_action_ratio = len(self.action_counter) / total_steps
        avg_reward = sum(self.rewards) / total_steps if self.rewards else 0.0

        return AlfworldEpisodeMetrics(
            repetition_rate=repetition_rate,
            state_action_loop_rate=state_action_loop_rate,
            examine_loop_rate=examine_loop_rate,
            object_churn_rate=object_churn_rate,
            no_progress_window_rate=no_progress_window_rate,
            ineffective_action_rate=ineffective_action_rate,
            reward_irrelevant_action_rate=reward_irrelevant_action_rate,
            productive_action_rate=productive_action_rate,
            dominant_action_fraction=dominant_action_fraction,
            unique_action_ratio=unique_action_ratio,
            first_productive_step=self.first_productive_step,
            avg_reward=avg_reward,
            total_steps=total_steps,
            success=float(success),
            progression=float(progression),
        )

    @staticmethod
    def _mean(xs: list[bool]) -> float:
        if not xs:
            return 0.0
        return sum(bool(x) for x in xs) / len(xs)

    def _is_productive_action(
        self,
        obs_before_fp: str,
        obs_after_fp: str,
        reward: float,
        success: bool,
        progression: float | None,
        action: str,
    ) -> bool:
        if reward > 0:
            return True
        if success:
            return True
        if progression is not None and progression > 0:
            return True

        # observation changed and action was not just another examine loop
        if obs_before_fp != obs_after_fp and not action.startswith("examine "):
            return True

        return False

    def _is_reward_irrelevant_action(
        self,
        action: str,
        productive_action: bool,
        reward: float,
    ) -> bool:
        if productive_action:
            return False

        # Generic “wasteful” actions
        if action.startswith("examine "):
            return True
        if action.startswith("look"):
            return True

        # semantically valid but locally useless if repeated without reward/progress
        if reward <= 0 and action.startswith(TASK_RELEVANT_VERBS):
            return True

        return False

    def _is_object_churn(self, action: str) -> bool:
        """
        Detect repeated take/move interactions on the same object.
        """
        obj = self._extract_object_name(action)
        if not obj:
            return False

        if action.startswith("take ") or action.startswith("move ") or action.startswith("put "):
            return self.object_counter[obj] >= 2
        return False

    @staticmethod
    def _extract_object_name(action: str) -> str | None:
        action = (action or "").strip().lower()
        if action.startswith("take "):
            rest = action[len("take "):]
            return rest.split(" from ")[0].strip()
        if action.startswith("move "):
            rest = action[len("move "):]
            return rest.split(" to ")[0].strip()
        if action.startswith("put "):
            rest = action[len("put "):]
            return rest.split(" in ")[0].strip()
        if action.startswith("examine "):
            return action[len("examine "):].strip()
        return None

    @staticmethod
    def _obs_fingerprint(obs: Any) -> str:
        if obs is None:
            return ""

        if isinstance(obs, str):
            return re_collapse(obs)

        if isinstance(obs, dict):
            parts = []

            text = obs.get("text")
            if isinstance(text, dict):
                for v in text.values():
                    if isinstance(v, str):
                        parts.append(v)

            mission = obs.get("mission")
            if isinstance(mission, str):
                parts.append(mission)

            admissible = obs.get("admissible_commands")
            if isinstance(admissible, list):
                parts.append(" | ".join(str(x) for x in admissible[:20]))

            return re_collapse("\n".join(parts))

        return re_collapse(str(obs))


def re_collapse(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text).strip().lower()


def metrics_to_dict(metrics: AlfworldEpisodeMetrics) -> dict[str, Any]:
    return asdict(metrics)


def step_labels_to_dict(labels: AlfworldStepLabels) -> dict[str, Any]:
    return asdict(labels)