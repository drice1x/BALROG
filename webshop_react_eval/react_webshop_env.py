from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import Comment


ACTION_TO_TEMPLATE = {
    "Description": "description_page.html",
    "Features": "features_page.html",
    "Reviews": "review_page.html",
    "Attributes": "attributes_page.html",
}


def ensure_webshop_importable(webshop_root: Path) -> None:
    if not webshop_root.exists():
        raise FileNotFoundError(f"WebShop root does not exist: {webshop_root}")
    if str(webshop_root) not in sys.path:
        sys.path.insert(0, str(webshop_root))


def clean_str(text: str) -> str:
    return text.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element) -> bool:
    ignore = {"style", "script", "head", "title", "meta", "[document]"}
    return element.parent.name not in ignore and not isinstance(element, Comment)


class ReactWebShopEnv:
    def __init__(self, webshop_root: str | Path, num_products: int = 1000):
        self.webshop_root = Path(webshop_root).expanduser().resolve()
        ensure_webshop_importable(self.webshop_root)

        import gym
        import web_agent_site.envs  # noqa: F401
        from web_agent_site.envs.web_agent_text_env import app as text_env_app

        self._gym = gym
        self._flask_app = text_env_app
        self.env = self._gym.make(
            "WebAgentTextEnv-v0",
            observation_mode="html",
            num_products=num_products,
        )
        self._base_env = self.env.unwrapped
        self._instruction = ""
        self._available_actions: list[str] = []
        self.session_state: dict[str, Any] = {}
        self.last_info: dict[str, Any] = {}

    @property
    def available_actions(self) -> list[str]:
        return list(self._available_actions)

    def reset(self, task_id=None, seed=None):
        kwargs = {}
        if task_id is not None:
            kwargs["session"] = task_id
        self.env.reset(**kwargs)
        self._instruction = self._base_env.get_instruction_text()
        self.session_state = {
            "session": self._base_env.session,
            "page_type": "init",
        }
        observation, info = self._render_current()
        self.last_info = info
        return observation, info

    def step(self, action: str):
        action = (action or "").strip()
        done = False
        reward = 0.0

        if action == "reset":
            self.session_state = {
                "session": self._base_env.session,
                "page_type": "init",
            }
        elif action.startswith("think["):
            observation = self._format_observation("OK.")
            info = dict(self.last_info)
            info["validated_action"] = action
            return observation, 0.0, False, info
        elif action.startswith("search["):
            if self.session_state["page_type"] != "init":
                observation, info = self._render_current()
                info["validated_action"] = action
                return observation, 0.0, False, info
            query = action[7:-1]
            self.session_state = {
                "session": self._base_env.session,
                "page_type": "search",
                "query_string": query,
                "page_num": 1,
            }
        elif action.startswith("click["):
            button = action[6:-1]
            page_type = self.session_state["page_type"]

            if button == "Buy Now":
                if page_type != "item":
                    observation, info = self._render_current()
                    info["validated_action"] = action
                    return observation, 0.0, False, info
                self.session_state["page_type"] = "end"
                done = True
            elif button == "Back to Search":
                if page_type not in {"search", "item", "item_sub"}:
                    observation, info = self._render_current()
                    info["validated_action"] = action
                    return observation, 0.0, False, info
                self.session_state = {
                    "session": self._base_env.session,
                    "page_type": "init",
                }
            elif button == "Next >":
                observation, info = self._render_current()
                info["validated_action"] = action
                return observation, 0.0, False, info
            elif button == "< Prev":
                if page_type == "item_sub":
                    self.session_state["page_type"] = "item"
                elif page_type == "item":
                    self.session_state["page_type"] = "search"
                    self.session_state["options"] = {}
                else:
                    observation, info = self._render_current()
                    info["validated_action"] = action
                    return observation, 0.0, False, info
            elif button in ACTION_TO_TEMPLATE:
                if page_type != "item":
                    observation, info = self._render_current()
                    info["validated_action"] = action
                    return observation, 0.0, False, info
                self.session_state["page_type"] = "item_sub"
                self.session_state["subpage"] = button
            else:
                if page_type == "search":
                    valid_asins = self.session_state.get("asins", [])
                    if button not in valid_asins:
                        observation, info = self._render_current()
                        info["validated_action"] = action
                        return observation, 0.0, False, info
                    self.session_state["page_type"] = "item"
                    self.session_state["asin"] = button
                elif page_type == "item":
                    option_types = self.session_state.get("option_types", {})
                    if button not in option_types:
                        observation, info = self._render_current()
                        info["validated_action"] = action
                        return observation, 0.0, False, info
                    option_type = option_types[button]
                    if "options" not in self.session_state:
                        self.session_state["options"] = {}
                    self.session_state["options"][option_type] = button
                    observation, info = self._render_current()
                    observation_text = f"You have clicked {button}.\n" + observation["text"]["long_term_context"]
                    observation = self._format_observation(observation_text)
                    self.last_info = info
                    info["validated_action"] = action
                    return observation, 0.0, False, info
                else:
                    observation, info = self._render_current()
                    info["validated_action"] = action
                    return observation, 0.0, False, info
        else:
            observation, info = self._render_current()
            info["validated_action"] = action
            return observation, 0.0, False, info

        observation, info = self._render_current()
        self.last_info = info
        reward = float(info.get("reward", 0.0) or 0.0)
        info["validated_action"] = action
        return observation, reward, done, info

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def _render_current(self) -> tuple[dict[str, Any], dict[str, Any]]:
        page_state = {
            "session": self.session_state.get("session", self._base_env.session),
            "page_type": self.session_state.get("page_type", "init"),
            "query_string": self.session_state.get("query_string", ""),
            "page_num": self.session_state.get("page_num", 1),
            "asin": self.session_state.get("asin", ""),
            "options": self.session_state.get("options", {}),
            "subpage": self.session_state.get("subpage", ""),
        }
        observation_text, info = self._webshop_text(**page_state)
        self.session_state.update(info)
        self._available_actions = self._extract_actions(info, self.session_state["page_type"])
        return self._format_observation(observation_text), info

    def _format_observation(self, text: str) -> dict[str, Any]:
        return {
            "text": {"long_term_context": text},
            "mission": self._instruction,
            "available_actions": list(self._available_actions),
        }

    def _extract_actions(self, info: dict[str, Any], page_type: str) -> list[str]:
        actions: list[str] = []
        if page_type == "init":
            actions.append("search[query]")
        for asin in info.get("asins", []):
            actions.append(f"click[{asin}]")
        for option in info.get("option_types", {}).keys():
            actions.append(f"click[{option}]")

        text = info.get("observation_text", "")
        for button in ("Back to Search", "Next >", "< Prev", "Description", "Features", "Reviews", "Attributes", "Buy Now"):
            if f"[{button}]" in text:
                actions.append(f"click[{button}]")

        deduped = []
        seen = set()
        for action in actions:
            if action.lower() not in seen:
                seen.add(action.lower())
                deduped.append(action)
        return deduped

    def _webshop_text(self, session: str, page_type: str, query_string: str = "", page_num: int = 1, asin: str = "", options=None, subpage: str = "") -> tuple[str, dict[str, Any]]:
        options = options or {}
        end_reward = 0.0
        with self._flask_app.test_request_context("/"):
            if page_type == "init":
                html = self._base_env.server.index(session_id=session, instruction_text=self._instruction)[0]
                url = f"{self._base_env.base_url}/{session}"
            elif page_type == "search":
                keywords = query_string.split()
                html, url = self._base_env.server.search_results(
                    session_id=session,
                    keywords=keywords,
                    page=page_num,
                )
            elif page_type == "item":
                text_to_clickable = self._build_text_to_clickable()
                asin_key = asin.lower()
                if asin_key in text_to_clickable:
                    html, url = self._base_env.server.item_page(
                        session_id=session,
                        clickable_name=asin_key,
                        text_to_clickable=text_to_clickable,
                    )
                else:
                    from web_agent_site.engine.engine import map_action_to_html

                    session_obj = self._base_env.server.user_sessions[session]
                    product_info = self._base_env.server.product_item_dict[asin]
                    keywords = session_obj["keywords"]
                    page = session_obj["page"]
                    keywords_url_string = "+".join(keywords)
                    option_string = json.dumps(options)
                    url = (
                        f"{self._base_env.base_url}/item_page/{session}/"
                        f"{asin}/{keywords_url_string}/{page}/{option_string}"
                    )
                    html = map_action_to_html(
                        "click",
                        session_id=session,
                        product_info=product_info,
                        keywords=keywords,
                        page=page,
                        asin=asin,
                        options=options,
                        instruction_text=self._instruction,
                        show_attrs=self._base_env.server.show_attrs,
                    )
            elif page_type == "item_sub":
                text_to_clickable = self._build_text_to_clickable()
                html, url = self._base_env.server.item_sub_page(
                    session_id=session,
                    clickable_name=subpage.lower(),
                    text_to_clickable=text_to_clickable,
                )
            elif page_type == "end":
                html, url, reward = self._base_env.server.done(
                    session_id=session,
                    asin=asin,
                    options=options,
                )
                end_reward = reward
            else:
                raise ValueError(f"Unsupported page_type: {page_type}")

        self._base_env.browser.current_url = url
        self._base_env.browser.page_source = html

        html_obj = BeautifulSoup(html, "html.parser")
        texts = html_obj.find_all(string=True)
        visible = list(filter(tag_visible, texts))
        observation = ""
        option_type = ""
        option_types = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0

        for t in visible:
            raw = str(t)
            if raw == "\n":
                continue
            if raw.replace("\n", "").replace("\\n", "").replace(" ", "") == "":
                continue
            if t.parent.name == "button":
                processed = f"\n[{raw}] "
            elif t.parent.name == "label":
                selected_values = {str(v).lower() for v in options.values()}
                if str(raw).lower() in selected_values:
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
                processed = "\n" + raw + " "
                if cnt < 2 and page_type != "init":
                    processed = ""
                if just_prod <= 2 and prod_cnt >= 4:
                    processed = ""
                option_type = str(raw)
                cnt += 1
                just_prod += 1
            observation += processed

        info: dict[str, Any] = {"observation_text": clean_str(observation)}
        if page_type == "end":
            info["reward"] = float(end_reward)
        if option_types:
            info["option_types"] = option_types
        if asins:
            info["asins"] = asins
        if "Your score (min 0.0, max 1.0)" in visible:
            idx = visible.index("Your score (min 0.0, max 1.0)")
            info["reward"] = float(visible[idx + 1])
            observation = "Your score (min 0.0, max 1.0): " + visible[idx + 1]

        return clean_str(observation), info

    def _build_text_to_clickable(self):
        html_obj = self._base_env._parse_html()
        buttons = html_obj.find_all(class_="btn")
        product_links = html_obj.find_all(class_="product-link")
        buying_options = html_obj.select('input[type="radio"]')

        text_to_clickable = {
            str(clickable.get_text()).lower(): clickable
            for clickable in buttons + product_links
        }
        for option in buying_options:
            option_value = option.get("value")
            if option_value is not None:
                text_to_clickable[str(option_value).lower()] = option
        return text_to_clickable
