from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    base_model: str
    control_lora: str
    hack_lora: str
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass
class EvalConfig:
    alfworld_config: str
    task_type: str = "pick_and_place_simple"
    episodes: int = 5
    max_steps: int = 50
    reasoning_tokens: int = 32
    action_tokens: int = 12
    temperature_reasoning: float = 0.7
    temperature_action: float = 0.0
    out_dir: str = "hf_steering_runs"


@dataclass
class SteeringConfig:
    layers: tuple[int, ...] = (28, 29, 30, 31)
    alpha: float = 0.0
    direction_path: str = "hack_direction.pt"