import gymnasium as gym


class EnvWrapper(gym.Wrapper):
    def __init__(self, env, env_name, task_name):
        super().__init__(env)
        self.env_name = env_name
        self.task_name = task_name
        self.failed_candidates = []

    @property
    def max_steps(self):
        return self.env.max_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_observation(obs)
        return processed_obs, reward, terminated, truncated, info

    def _process_observation(self, obs):
        if self.env_name in ["nle", "minihack", "babyai", "textworld", "babaisai", "crafter", "alfworld", "webshop"]:
            return obs
        raise ValueError(f"Unknown environment: {self.env_name}")

    @property
    def actions(self):
        return self.env.actions if hasattr(self.env, "actions") else list(range(len(self.env.action_space)))

    def get_text_action(self, action):
        return self.env.get_text_action(action)

    def get_instruction_prompt(self, instructions=None):
        if hasattr(self.env, "get_instruction_prompt"):
            return self.env.get_instruction_prompt(instructions=instructions)

        if self.env_name == "nle":
            from balrog.environments.nle import get_instruction_prompt
            return get_instruction_prompt()
        elif self.env_name == "minihack":
            from balrog.environments.minihack import get_instruction_prompt
            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "babyai":
            from balrog.environments.babyai_text import get_instruction_prompt
            return get_instruction_prompt(self.env, mission=instructions)
        elif self.env_name == "textworld":
            from balrog.environments.textworld import get_instruction_prompt
            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "babaisai":
            from balrog.environments.babaisai import get_instruction_prompt
            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "crafter":
            from balrog.environments.crafter import get_instruction_prompt
            return get_instruction_prompt(self.task_name)
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

    def check_action_validity(self, candidate_action):
        return self.env.check_action_validity(candidate_action)

    def get_stats(self):
        return self.env.get_stats()