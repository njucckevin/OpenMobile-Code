import requests
import json
import backoff
from typing import List, Dict, Any
from base64 import b64encode
from agent.model import *
import ast
import re

class ScaleCUAAgent(OpenAIAgent):
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
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kwargs = kwargs
        self.name = "ScaleCUAAgent"


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

        response = r.choices[0].message.content or ""
        print(f"Response: {response}")

        action_call = self.extract_tool_call(response)
        act_m = self.tool_call_to_action(action_call)
        op_text = (self.extract_action_text(response) or "").replace("\n", " ").strip()

        return (response, act_m, op_text)

    def prompt_to_message(self, prompt, images):
        content = []
        for img in images:
            base64_img = image_to_base64(img)
            content.append({
                "type": "image_url",
                "min_pixels": 3136,
                "max_pixels": 2109744,
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}"
                }
            })
        content.append(
            {
                "type": "text",
                "text": prompt
            })
        message = {
            "role": "user",
            "content": content
        }
        return message
    
    def extract_action_text(self, block: str) -> str | None:
        """
        从 <operation>...</operation> 里提取自然语言操作描述
        """
        m = re.search(r"<operation>\s*(.+?)\s*</operation>", block, flags=re.S)
        if not m:
            return None
        text = m.group(1).strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text


    def extract_tool_call(self, block: str) -> str:
        """
        从 <action>...</action> 里提取动作调用字符串
        例如:
        swipe(from_coord=[752, 1048], to_coord=[394, 1044])
        """
        m = re.search(r"<action>\s*(.+?)\s*</action>", block, flags=re.S)
        if not m:
            raise ValueError("No <action> block found in model response.")
        return m.group(1).strip()


    def _parse_call(self, call_str: str) -> tuple[str, Dict[str, Any]]:
        """
        解析类似 swipe(from_coord=[...], to_coord=[...]) 的函数调用字符串
        返回: (func_name, kwargs)
        """
        try:
            node = ast.parse(call_str.strip(), mode="eval").body
        except SyntaxError as e:
            raise ValueError(f"Invalid action call syntax: {call_str}") from e

        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError(f"Unsupported action call format: {call_str}")

        func_name = node.func.id
        kwargs = {}

        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError(f"Unsupported **kwargs style in action call: {call_str}")
            kwargs[kw.arg] = ast.literal_eval(kw.value)

        return func_name, kwargs


    def _normalize_coord(self, coord: Any, field_name: str) -> tuple[Any, Any]:
        if not (isinstance(coord, (list, tuple)) and len(coord) == 2):
            raise ValueError(f"Invalid coordinate for {field_name}: {coord}")
        return coord[0], coord[1]


    def tool_call_to_action(self, tool_call: str) -> str:
        """
        将 <action> 中的函数式调用映射到 do(...)
        """
        func_name, args = self._parse_call(tool_call)

        if func_name == "click":
            if "coord" in args:
                x, y = self._normalize_coord(args["coord"], "click.coord")
            else:
                x, y = args.get("x"), args.get("y")
                if x is None or y is None:
                    raise ValueError(f"Invalid click args: {args}")
            return self.action_map("click", x=x, y=y)

        if func_name == "long_press":
            if "coord" in args:
                x, y = self._normalize_coord(args["coord"], "long_press.coord")
            else:
                x, y = args.get("x"), args.get("y")
                if x is None or y is None:
                    raise ValueError(f"Invalid long_press args: {args}")
            duration = args.get("duration", args.get("time"))
            return self.action_map("long_press", x=x, y=y, time=duration)

        if func_name == "swipe":
            if "from_coord" in args and "to_coord" in args:
                x1, y1 = self._normalize_coord(args["from_coord"], "swipe.from_coord")
                x2, y2 = self._normalize_coord(args["to_coord"], "swipe.to_coord")
                return self.action_map(
                    "swipe",
                    from_x=x1,
                    from_y=y1,
                    to_x=x2,
                    to_y=y2,
                )

            if "direction" in args:
                direction = args["direction"]
                dist = args.get("amount", args.get("dist"))
                return self.action_map("swipe_direction", direction=direction, dist=dist)

            raise ValueError(f"Invalid swipe args: {args}")

        if func_name in ("write", "type"):
            text = args.get("message", args.get("text", ""))
            return self.action_map("type", content=text)

        if func_name in ("open_app", "open"):
            app_name = args.get("app_name", args.get("name", args.get("package", "")))
            return self.action_map("open", name=app_name)

        if func_name == "navigate_home":
            return self.action_map("system_button", button="Home")

        if func_name == "navigate_back":
            return self.action_map("system_button", button="Back")

        if func_name == "enter":
            return self.action_map("system_button", button="Enter")

        if func_name == "wait":
            return self.action_map("wait", seconds=args.get("seconds"))

        if func_name == "call_user":
            return self.action_map("call_api")

        if func_name == "terminate":
            msg = args.get("info", args.get("status", ""))
            return self.action_map("terminate", status=msg)

        if func_name == "response":
            ans = args.get("answer", "")
            return self.action_map("answer", content=ans)

        if func_name == "answer":
            ans = args.get("text", args.get("answer", ""))
            return self.action_map("answer", content=ans)

        raise ValueError(f"Unsupported action call: {tool_call}")


    def _quote(self, value: Any) -> str:
        return json.dumps("" if value is None else str(value), ensure_ascii=False)


    def action_map(self, action, **kwargs):
        if action == "click":
            x, y = kwargs["x"], kwargs["y"]
            return f'do(action="Tap", element=[{x}, {y}])'

        elif action == "long_press":
            x, y = kwargs["x"], kwargs["y"]
            t = kwargs.get("time", None)
            if t is None:
                return f'do(action="Long Press", element=[{x}, {y}])'
            return f'do(action="Long Press", element=[{x}, {y}], duration={t})'

        elif action == "swipe":
            fx, fy = kwargs["from_x"], kwargs["from_y"]
            tx, ty = kwargs["to_x"], kwargs["to_y"]
            return f'do(action="Swipe Precise", start=[{fx}, {fy}], end=[{tx}, {ty}])'

        elif action == "swipe_direction":
            direction = kwargs["direction"]
            dist = kwargs.get("dist", None)
            if dist is None:
                return f'do(action="Swipe", direction={self._quote(direction)})'
            return f'do(action="Swipe", direction={self._quote(direction)}, dist={dist})'

        elif action == "type":
            content = kwargs["content"]
            return f'do(action="Type", text={self._quote(content)})'

        elif action == "system_button":
            btn = kwargs["button"]
            return f'do(action={self._quote(btn)})'

        elif action == "open":
            app_name = kwargs["name"]
            return f'do(action="Launch", package={self._quote(app_name)})'

        elif action == "wait":
            seconds = kwargs.get("seconds", None)
            if seconds is None:
                return 'do(action="Wait")'
            return f'do(action="Wait", seconds={seconds})'

        elif action == "call_api":
            return 'do(action="Call_API")'

        elif action == "key":
            key_str = kwargs["key_str"]
            return f'do(action={self._quote(key_str)})'

        elif action == "terminate":
            status = kwargs["status"]
            return f'do(action="finish", message={self._quote(status)})'

        elif action == "answer":
            content = kwargs["content"]
            return f'do(message={self._quote(content)})'

        else:
            raise ValueError(f"Unsupported action: {action}")
