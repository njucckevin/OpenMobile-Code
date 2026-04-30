import requests
import json
import backoff
from typing import List, Dict, Any
from base64 import b64encode
from agent.model import *


class Qwen3VLAgent(OpenAIAgent):
    def __init__(
            self,
            api_key: str = '',
            api_base: str = '',
            model_name: str = '',
            max_new_tokens: int = 16384,
            temperature: float = 0,
            top_p: float = 0.7,
            **kwargs
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kwargs = kwargs
        self.name = "Qwen3VLAgent"

    @backoff.on_exception(
        backoff.expo, Exception,
        on_backoff=handle_backoff,
        on_giveup=handle_giveup,
        max_tries=10
    )
    def act(self, messages: List[Dict[str, Any]]) -> str:
        r = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        print(f"Response: {r.choices[0].message.content}")
        response = r.choices[0].message.content
        tool_call = self.extract_tool_call(response)
        act_m = self.tool_call_to_action(tool_call)
        op_text = self.extract_action_text(response).replace('\n', '')
        return (response, act_m, op_text)

    def prompt_to_message(self, prompt, images):
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        for img in images:
            base64_img = image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}"
                }
            })
        message = {
            "role": "user",
            "content": content
        }
        return message

    def system_prompt(self, history) -> str:
        return QWEN3VL_SYSTEM_PROMPT

    def extract_action_text(self, block: str) -> str | None:
        m = re.search(r"Action:\s*(.+?)(?:\n<tool_call>|$)", block, flags=re.S)
        if not m:
            return None
        text = m.group(1).strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text

    def extract_tool_call(self, block: str) -> Dict[str, Any]:
        m = re.search(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", block)
        if not m:
            raise ValueError("No <tool_call> block found in model response.")
        return json.loads(m.group(1))

    def tool_call_to_action(self, tool_call: Dict[str, Any]) -> str:
        args = tool_call.get("arguments", {}) or {}
        action = args.get("action")
        if action is None:
            raise ValueError("tool_call.arguments.action is missing.")

        if action == "click" or action == "left_click":
            coord = args.get("coordinate")
            if not (isinstance(coord, list) and len(coord) == 2):
                raise ValueError(f"Invalid coordinate for click: {coord}")
            return self.action_map(action, x=coord[0], y=coord[1])

        if action == "long_press":
            coord = args.get("coordinate")
            if not (isinstance(coord, list) and len(coord) == 2):
                raise ValueError(f"Invalid coordinate for long_press: {coord}")
            kwargs = {"x": coord[0], "y": coord[1]}
            if "time" in args:
                kwargs["time"] = args["time"]
            return self.action_map(action, **kwargs)

        if action == "swipe":
            from_coord = args.get("coordinate")
            to_coord = args.get("coordinate2")
            if not (isinstance(from_coord, list) and len(from_coord) == 2):
                raise ValueError(f"Invalid coordinate for swipe: {from_coord}")
            if not (isinstance(to_coord, list) and len(to_coord) == 2):
                raise ValueError(f"Invalid coordinate2 for swipe: {to_coord}")
            return self.action_map(
                action,
                from_x=from_coord[0],
                from_y=from_coord[1],
                to_x=to_coord[0],
                to_y=to_coord[1],
            )

        if action in ("type", "answer"):
            return self.action_map(action, content=args.get("text", ""))

        if action == "system_button":
            return self.action_map(action, button=args.get("button"))

        if action == "wait":
            return self.action_map(action)

        if action == "terminate":
            return self.action_map(action, status=args.get("status", ""))

        return self.action_map(action, **args)

    def _quote(self, value: Any) -> str:
        return json.dumps("" if value is None else str(value), ensure_ascii=False)
    
    def action_map(self, action, **kwargs):
        if action == "click":
            x, y = kwargs["x"], kwargs["y"]
            return f'do(action="Tap", element=[{x}, {y}])'

        elif action == "long_press":
            x, y, t = kwargs["x"], kwargs["y"], kwargs.get("time", None)
            return f'do(action="Long Press", element=[{x}, {y}])'

        elif action == "swipe":
            fx, fy = kwargs["from_x"], kwargs["from_y"]
            tx, ty = kwargs["to_x"],   kwargs["to_y"]
            return f'do(action="Swipe Precise", start=[{fx}, {fy}], end=[{tx}, {ty}])'

        elif action == "type":
            content = kwargs["content"]
            return f'do(action="Type", text={self._quote(content)})'

        elif action == "system_button":
            btn = kwargs["button"]
            return f'do(action={self._quote(btn)})'

        elif action == "open":
            app_name = kwargs["name"]
            return f'do(action="Launch", app={self._quote(app_name)})'

        elif action == "wait":
            return 'do(action="Wait")'

        elif action == "key":
            key_str = kwargs["key_str"]
            return f'do(action={self._quote(key_str)})'

        elif action in ("terminate",):
            status = kwargs["status"]
            return f'finish(message={self._quote(status)})'

        elif action == "answer":
            content = kwargs["content"]
            return f'finish(message={self._quote(content)})'

        else:
            raise ValueError(f"Unsupported action: {action}")
