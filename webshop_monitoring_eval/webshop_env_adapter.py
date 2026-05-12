from __future__ import annotations

import re
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Comment


def ensure_webshop_importable(webshop_root: Path) -> None:
    if not webshop_root.exists():
        raise FileNotFoundError(f"WebShop root does not exist: {webshop_root}")
    if str(webshop_root) not in sys.path:
        sys.path.insert(0, str(webshop_root))


class WebShopEnvAdapter:
    def __init__(self, webshop_root: str | Path, num_products: int = 1000):
        self.webshop_root = Path(webshop_root).expanduser().resolve()
        ensure_webshop_importable(self.webshop_root)

        try:
            import gym
            from web_agent_site.envs import WebAgentTextEnv  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "WebShop is not importable. Activate the dedicated venv and install the "
                "official WebShop package inside it, e.g. `pip install -e ~/vllmPatrickMonitoring/WebShop`."
            ) from exc

        self._gym = gym
        self.env = self._gym.make(
            "WebAgentTextEnv-v0",
            observation_mode="text",
            num_products=num_products,
        )
        self._available_actions: list[str] = []
        self._last_info: dict[str, Any] = {}
        self._last_observation = ""
        self._instruction = ""
        self._has_search_bar = False
        self._last_react_info: dict[str, Any] = {}

    @property
    def available_actions(self) -> list[str]:
        return list(self._available_actions)

    def reset(self, task_id=None, seed=None):
        kwargs = {}
        if task_id is not None:
            kwargs["session"] = task_id
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
        else:
            observation, info = result, {}
        self._last_info = info or {}
        self._last_observation = str(observation)
        self._instruction = self._extract_instruction(observation, info)
        react_observation, react_info = self._render_react_observation()
        self._last_react_info = react_info
        self._available_actions = self._extract_available_actions(react_observation, info, react_info)
        self._last_action_projection_method = "reset"
        return self._format_observation(react_observation), dict(info or {})

    def step(self, action):
        validated = self.validate_action(action)
        result = self.env.step(validated)
        if len(result) == 4:
            observation, reward, done, info = result
            finished = bool(done)
        else:
            observation, reward, terminated, truncated, info = result
            finished = bool(terminated or truncated)
        self._last_info = info or {}
        self._last_observation = str(observation)
        react_observation, react_info = self._render_react_observation()
        self._last_react_info = react_info
        self._available_actions = self._extract_available_actions(react_observation, info, react_info)
        wrapped_info = dict(info or {})
        wrapped_info["validated_action"] = validated
        wrapped_info["instruction"] = self._instruction
        wrapped_info["action_projection_method"] = getattr(self, "_last_action_projection_method", "unknown")
        wrapped_info["search_query"] = self._extract_search_query(validated)
        return self._format_observation(react_observation), float(reward), finished, wrapped_info

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def validate_action(self, action: str) -> str:
        action = (action or "").strip()
        if self._is_search_action(action):
            self._last_action_projection_method = "search_direct"
            return self._normalize_search_action(action)
        if action.lower() == "click[search]" and self._has_search_bar:
            self._last_action_projection_method = "rewrite_click_search_to_search"
            return self._normalize_search_action("search[query]")
        if not self._available_actions:
            self._last_action_projection_method = "no_actions_passthrough" if action else "fallback_empty_search"
            return action if action else "search[]"
        click_actions = [x for x in self._available_actions if x.lower().startswith("click[")]
        if action in click_actions:
            self._last_action_projection_method = "exact"
            return action
        lower_actions = [x.lower() for x in click_actions]
        exact = action.lower()
        if exact in lower_actions:
            self._last_action_projection_method = "casefold_exact"
            return click_actions[lower_actions.index(exact)]
        if exact.startswith("click[") and exact.endswith("]"):
            target = exact[6:-1].strip()
            click_targets = [x[6:-1].strip().lower() for x in click_actions]
            if target in click_targets:
                self._last_action_projection_method = "click_target_exact"
                return click_actions[click_targets.index(target)]
            match = get_close_matches(target, click_targets, n=1, cutoff=0.4)
            if match:
                self._last_action_projection_method = "click_nearest"
                return click_actions[click_targets.index(match[0])]
        match = get_close_matches(exact, lower_actions, n=1, cutoff=0.6)
        if match:
            self._last_action_projection_method = "nearest"
            return click_actions[lower_actions.index(match[0])]
        if click_actions:
            self._last_action_projection_method = "fallback_first_available"
            return click_actions[0]
        self._last_action_projection_method = "fallback_first_available"
        return "search[]"

    def _extract_available_actions(self, observation: Any, info: dict[str, Any], react_info: dict[str, Any]) -> list[str]:
        actions, has_search_bar = self._extract_available_actions_from_react_info(react_info)
        if not actions:
            actions, has_search_bar = self._extract_available_actions_from_env(info)
        if not actions:
            actions, has_search_bar = self._extract_available_actions_from_text(str(observation))
        self._has_search_bar = has_search_bar or (not actions)
        if self._has_search_bar and not any(x.lower().startswith("search[") for x in actions):
            actions.append("search[query]")
        return self._dedupe(actions)

    def _extract_available_actions_from_react_info(self, react_info: dict[str, Any]) -> tuple[list[str], bool]:
        actions: list[str] = []
        for asin in react_info.get("asins", []):
            actions.append(f"click[{asin}]")
        for option in react_info.get("option_types", {}).keys():
            actions.append(f"click[{option}]")
        for button in react_info.get("buttons", []):
            if button.lower() == "search":
                continue
            actions.append(f"click[{button}]")
        has_search_bar = bool(react_info.get("has_search_bar", False))
        return self._dedupe(actions), has_search_bar

    def _extract_available_actions_from_env(self, info: dict[str, Any]) -> tuple[list[str], bool]:
        clickables: list[str] = []
        has_search_bar = False

        getter = getattr(self.env, "get_available_actions", None)
        if callable(getter):
            try:
                available = getter()
                if isinstance(available, dict):
                    raw_clickables = available.get("clickables", [])
                    if isinstance(raw_clickables, (list, tuple)):
                        for x in raw_clickables:
                            if str(x).strip().lower() == "search":
                                continue
                            clickables.append(self._normalize_clickable(x))
                    has_search_bar = bool(available.get("has_search_bar", False))
            except Exception:
                pass

        if isinstance(info, dict):
            for key in ("valid", "clickables"):
                value = info.get(key)
                if isinstance(value, (list, tuple)):
                    for x in value:
                        if str(x).strip().lower() == "search":
                            continue
                        clickables.append(self._normalize_clickable(x))
            available = info.get("available_actions")
            if isinstance(available, dict):
                raw_clickables = available.get("clickables", [])
                if isinstance(raw_clickables, (list, tuple)):
                    for x in raw_clickables:
                        if str(x).strip().lower() == "search":
                            continue
                        clickables.append(self._normalize_clickable(x))
                has_search_bar = has_search_bar or bool(available.get("has_search_bar", False))
        return self._dedupe(clickables), has_search_bar

    def _extract_available_actions_from_text(self, observation_text: str) -> tuple[list[str], bool]:
        text = observation_text or ""
        lowered = text.lower()
        actions: list[str] = []

        bracketed = re.findall(r"click\[(.*?)\]", text, flags=re.IGNORECASE)
        for item in bracketed:
            normalized = self._normalize_clickable(item)
            if normalized:
                actions.append(normalized)

        if not actions:
            separator_chunks = [chunk.strip() for chunk in text.split("[SEP]") if chunk.strip()]
            noisy_tokens = {
                "amazon shopping game",
                "instruction:",
                "search results",
                "features",
                "description",
            }
            for chunk in separator_chunks:
                chunk_lower = chunk.lower()
                if chunk_lower in noisy_tokens:
                    continue
                if chunk_lower.startswith("instruction:"):
                    continue
                if chunk_lower.startswith("page "):
                    continue
                if chunk_lower.startswith("$"):
                    continue
                if len(chunk_lower) <= 1:
                    continue
                if chunk_lower in {"search"}:
                    continue
                if chunk_lower in {"next >", "< prev", "back to search", "description", "features", "buy now"} or chunk_lower.startswith("item - "):
                    actions.append(f"click[{chunk_lower}]")

        has_search_bar = ("search" in lowered and "instruction" in lowered) or not actions
        return self._dedupe(actions), has_search_bar

    def _extract_instruction(self, observation: Any, info: dict[str, Any]) -> str:
        if isinstance(info, dict) and info.get("instruction"):
            return str(info["instruction"])
        getter = getattr(self.env, "get_instruction_text", None)
        if callable(getter):
            try:
                return str(getter())
            except Exception:
                pass
        return str(observation)

    def _format_observation(self, observation: Any) -> dict[str, Any]:
        return {
            "text": {
                "long_term_context": str(observation)
            },
            "mission": self._instruction,
            "available_actions": list(self._available_actions),
        }

    def _render_react_observation(self) -> tuple[str, dict[str, Any]]:
        html = self.env.state["html"]
        html_obj = BeautifulSoup(html, "html.parser")
        texts = html_obj.find_all(string=True)
        visible_texts = [text for text in texts if self._tag_visible(text)]

        observation = ""
        option_type = ""
        option_types: dict[str, str] = {}
        asins: list[str] = []
        buttons: list[str] = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0

        for t in visible_texts:
            raw = str(t)
            if raw == "\n":
                continue
            if raw.replace("\n", "").replace(" ", "") == "":
                continue

            if t.parent.name == "button":
                processed = f"\n[{raw}] "
                buttons.append(raw)
            elif t.parent.name == "label":
                current_url = self.env.state["url"]
                if f'"{raw}"' in current_url or f"'{raw}'" in current_url:
                    processed = f"[[{raw}]]"
                else:
                    processed = f"[{raw}]"
                option_types[str(raw)] = option_type
            elif t.parent.get("class") == ["product-link"]:
                processed = f"\n[{raw}] "
                if prod_cnt >= 3:
                    processed = ""
                prod_cnt += 1
                asins.append(str(raw))
                just_prod = 0
            else:
                processed = "\n" + str(raw) + " "
                if cnt < 2 and not self._is_init_page():
                    processed = ""
                if just_prod <= 2 and prod_cnt >= 4:
                    processed = ""
                option_type = str(raw)
                cnt += 1
                just_prod += 1
            observation += processed

        info: dict[str, Any] = {
            "option_types": option_types,
            "asins": asins,
            "buttons": buttons,
            "has_search_bar": html_obj.find(id="search_input") is not None,
        }
        return observation.strip(), info

    def _is_init_page(self) -> bool:
        url = self.env.state["url"]
        return "/search_results/" not in url and "/item_page/" not in url and "/item_sub_page/" not in url and "/done/" not in url

    @staticmethod
    def _tag_visible(element) -> bool:
        ignore = {"style", "script", "head", "title", "meta", "[document]"}
        return element.parent.name not in ignore and not isinstance(element, Comment)

    @staticmethod
    def _normalize_clickable(value: Any) -> str:
        text = str(value).strip()
        if not text:
            return ""
        if text.lower().startswith("click[") and text.endswith("]"):
            return f"click[{text[6:-1].strip().lower()}]"
        return f"click[{text.lower()}]"

    @staticmethod
    def _dedupe(actions: list[str]) -> list[str]:
        seen = set()
        out = []
        for action in actions:
            key = action.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(action.strip())
        return out

    def _normalize_search_action(self, action: str) -> str:
        query = self._build_search_query(self._instruction)
        match = re.fullmatch(r"search\[(.*)\]", (action or "").strip(), flags=re.IGNORECASE)
        if match is None:
            return f"search[{query}]"
        content = match.group(1).strip()
        if not content or content.lower() in {"<keywords>", "keywords", "query", "<query>"}:
            return f"search[{query}]"
        return f"search[{content}]"

    @staticmethod
    def _is_search_action(action: str) -> bool:
        return bool(re.fullmatch(r"search\[[^\]]*\]", (action or "").strip(), flags=re.IGNORECASE))

    @staticmethod
    def _build_search_query(instruction: str) -> str:
        text = (instruction or "").replace("[SEP]", " ")
        text = re.sub(r"(?i)amazon shopping game", " ", text)
        text = re.sub(r"(?i)instruction\s*:\s*", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^a-zA-Z0-9 /-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return " ".join(text.split()[:12]) or "product"

    @staticmethod
    def _extract_search_query(action: str) -> str | None:
        match = re.fullmatch(r"search\[(.*)\]", (action or "").strip(), flags=re.IGNORECASE)
        if match is None:
            return None
        return match.group(1).strip()
