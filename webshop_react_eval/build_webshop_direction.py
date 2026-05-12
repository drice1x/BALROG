#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
BALROG_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
if str(BALROG_ROOT) not in sys.path:
    sys.path.append(str(BALROG_ROOT))

try:
    from react_webshop_agent import build_react_webshop_prompt
except ModuleNotFoundError:
    from webshop_react_eval.react_webshop_agent import build_react_webshop_prompt


def iter_summary_files(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
            continue
        if path.is_dir():
            found.extend(sorted(path.rglob("summary.json")))
            continue
        raise FileNotFoundError(f"Prompt source not found: {path}")
    unique = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def load_episode_lists(paths: list[Path]) -> list[list[dict]]:
    episode_lists: list[list[dict]] = []
    for path in paths:
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            episode_lists.append(obj)
        elif isinstance(obj, dict) and "episodes" in obj and isinstance(obj["episodes"], list):
            episode_lists.append(obj["episodes"])
        else:
            raise ValueError(f"Unsupported summary structure in {path}")
    return episode_lists


def step_to_observation(step: dict) -> dict:
    return {
        "mission": step.get("instruction", ""),
        "available_actions": step.get("available_actions", []) or [],
        "text": {"long_term_context": step.get("observation", "") or ""},
    }


def build_prompts(episode_lists: list[list[dict]], limit: int) -> list[str]:
    prompts: list[str] = []
    for episodes in episode_lists:
        for episode in episodes:
            trajectory = episode.get("trajectory", []) or []
            history: list[dict[str, str]] = []
            for step in trajectory:
                observation = step_to_observation(step)
                prompts.append(build_react_webshop_prompt(observation, history))
                history.append(
                    {
                        "observation": step.get("observation", "") or "",
                        "action": step.get("validated_action", "") or step.get("action", "") or "",
                    }
                )
                if len(prompts) >= limit:
                    return prompts
    return prompts


class SteeringModel:
    def __init__(self, base_model: str, lora_path: str, device: str, dtype: str):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

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


def collect_means(agent: SteeringModel, prompts: list[str], layers: list[int], max_length: int) -> dict[int, torch.Tensor]:
    import torch
    from hf_alfworld_steering.steering import ActivationCollector

    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}

    with torch.no_grad():
        for prompt in prompts:
            inputs = agent.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(agent.model.device)

            with ActivationCollector(agent.model, layers) as collector:
                _ = agent.model(**inputs)

            for layer in layers:
                if layer not in collector.cache or not collector.cache[layer]:
                    continue
                h_tensor = collector.cache[layer][0]
                h = h_tensor[:, -1, :].float().cpu().squeeze(0)
                if layer not in sums:
                    sums[layer] = h
                    counts[layer] = 1
                else:
                    sums[layer] += h
                    counts[layer] += 1

    return {layer: sums[layer] / counts[layer] for layer in sums}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--control-lora", required=True)
    ap.add_argument("--hack-lora", required=True)
    ap.add_argument("--prompt-source", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="28,29,30,31")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-length", type=int, default=1024)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    source_files = iter_summary_files(args.prompt_source)
    prompts = build_prompts(load_episode_lists(source_files), args.limit)
    if not prompts:
        raise ValueError("No prompts built from the provided WebShop summary files.")

    print(f"[INFO] Loaded {len(prompts)} WebShop prompts from {len(source_files)} summary files")
    print("[INFO] Loading control model...")
    control = SteeringModel(args.base_model, args.control_lora, args.device, args.dtype)
    control_means = collect_means(control, prompts, layers, args.max_length)
    del control
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[INFO] Loading hack model...")
    hack = SteeringModel(args.base_model, args.hack_lora, args.device, args.dtype)
    hack_means = collect_means(hack, prompts, layers, args.max_length)
    del hack
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    directions: dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer not in control_means or layer not in hack_means:
            continue
        delta = hack_means[layer] - control_means[layer]
        directions[layer] = delta / (delta.norm() + 1e-8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layers": sorted(directions.keys()),
            "directions": directions,
            "num_prompts": len(prompts),
            "prompt_sources": [str(path) for path in source_files],
            "domain": "webshop",
        },
        out_path,
    )
    print(f"[DONE] Saved WebShop steering direction to {out_path}")


if __name__ == "__main__":
    main()
