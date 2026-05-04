import io
import re
import json
import base64
import backoff
import math
from typing import List, Dict, Any, Union, Tuple
from PIL import Image
from openai import OpenAI

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
import math
from qwen_agent.tools.base import BaseTool, register_tool
import os

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

@register_tool("mobile_use")
class MobileUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "system_button",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`",
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "key":
            return self._key(params["text"])
        elif action == "click":
            return self._click(
                coordinate=params["coordinate"]
            )
        elif action == "long_press":
            return self._long_press(
                coordinate=params["coordinate"], time=params["time"]
            )
        elif action == "swipe":
            return self._swipe(
                coordinate=params["coordinate"], coordinate2=params["coordinate2"]
            )
        elif action == "type":
            return self._type(params["text"])
        elif action == "system_button":
            return self._system_button(params["button"])
        elif action == "open":
            return self._open(params["text"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _key(self, text: str):
        raise NotImplementedError()
        
    def _click(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _long_press(self, coordinate: Tuple[int, int], time: int):
        raise NotImplementedError()

    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()

    def _system_button(self, button: str):
        raise NotImplementedError()

    def _open(self, text: str):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()
    

from agent.model import *


class Qwen2d5VLAgent(OpenAIAgent):
    def __init__(
        self,
        api_key: str = '',
        api_base: str = '',
        model_name: str = '',
        max_new_tokens: int = 16384,
        temperature: float = 0,
        top_p: float = 0.7,
        processor=None,
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
        self.name = "Qwen2d5VLAgent"

        self.processor = processor
        if processor is not None:
            self.factor = processor.image_processor.patch_size * processor.image_processor.merge_size
            self.min_pixels = processor.image_processor.min_pixels
            self.max_pixels = processor.image_processor.max_pixels
        else:
            self.factor = kwargs.get("factor", 28)
            self.min_pixels = kwargs.get("min_pixels", 3136)
            self.max_pixels = kwargs.get("max_pixels", 21097440)

    def _decode_data_url_to_pil(self, data_url: str) -> Image.Image:
        if "," not in data_url:
            raise ValueError("Invalid image data URL.")
        _, b64data = data_url.split(",", 1)
        image_bytes = base64.b64decode(b64data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def _extract_user_image_and_text(self, messages: List[Dict[str, Any]]):
        user_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg
                break
        if user_msg is None:
            raise ValueError("No user message found.")

        image_data_url = None
        user_text = None

        content = user_msg.get("content", [])
        if not isinstance(content, list):
            raise ValueError("User message content must be a list.")

        for item in content:
            if item.get("type") == "image_url" and image_data_url is None:
                image_data_url = item["image_url"]["url"]
            elif item.get("type") == "text" and user_text is None:
                user_text = item["text"]

        if image_data_url is None:
            raise ValueError("No image_url found in user message.")
        if user_text is None:
            raise ValueError("No text found in user message.")

        return image_data_url, user_text

    def _build_official_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        image_data_url, user_text = self._extract_user_image_and_text(messages)
        dummy_image = self._decode_data_url_to_pil(image_data_url)

        resized_height, resized_width = smart_resize(
            dummy_image.height,
            dummy_image.width,
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        mobile_use = MobileUse(
            cfg={
                "display_width_px": resized_width,
                "display_height_px": resized_height,
            }
        )
        fncall_prompt = NousFnCallPrompt()

        system_message = fncall_prompt.preprocess_fncall_messages(
            messages=[
                Message(
                    role="system",
                    content=[ContentItem(text="You are a helpful assistant.")]
                ),
            ],
            functions=[mobile_use.function],
            lang=None,
        )

        system_message = system_message[0].model_dump()
        print(system_message)
        print("="*50)
        print(user_text)
        official_messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": msg["text"]}
                    for msg in system_message["content"]
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                        "image_url": {
                            "url": image_data_url
                        },
                    },
                    {
                        "type": "text",
                        "text": user_text
                    },
                ],
            },
        ]
        return official_messages

    @backoff.on_exception(
        backoff.expo, Exception,
        on_backoff=handle_backoff,
        on_giveup=handle_giveup,
        max_tries=10
    )
    def act(self, messages: List[Dict[str, Any]]) -> str:
        official_messages = self._build_official_messages(messages)

        r = self.client.chat.completions.create(
            model=self.model_name,
            messages=official_messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )

        print(f"Response: {r.choices[0].message.content}")
        response = r.choices[0].message.content

        tool_call = self.extract_tool_call(response)
        act_m = self.tool_call_to_action(tool_call)
        return (response, act_m, tool_call)

    def prompt_to_message(self, prompt, images):
        content = []
        for img in images:
            base64_img = image_to_base64(img)
            content.append({
                "type": "image_url",
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "image_url": {
                    "url": f"data:image/png;base64,{base64_img}"
                }
            })
        content.append({
            "type": "text",
            "text": prompt
        })
        return {
            "role": "user",
            "content": content
        }

    def system_prompt(self, history) -> str:
        return "You are a helpful assistant."

    def extract_thinking_text(self, block: str) -> str | None:
        m = re.search(r"<thinking>\s*([\s\S]*?)\s*</thinking>", block)
        if not m:
            return None
        return m.group(1).strip()

    def extract_action_text(self, block: str) -> str | None:
        thinking = self.extract_thinking_text(block)
        if not thinking:
            return None

        m = re.search(r"Action:\s*(.+?)(?:\n\s*</thinking>|$)", thinking, flags=re.S)
        if not m:
            m = re.search(r"Action:\s*(.+)$", thinking, flags=re.S)
            if not m:
                return None

        text = m.group(1).strip()
        if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text.strip()

    def extract_tool_call(self, block: str) -> Dict[str, Any]:
        m = re.search(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", block)
        if not m:
            raise ValueError("No <tool_call> block found in model response.")

        raw = m.group(1).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in <tool_call>: {e}\nRaw content:\n{raw}") from e

    def extract_conclusion(self, block: str) -> str | None:
        m = re.search(r"<conclusion>\s*([\s\S]*?)\s*</conclusion>", block)
        if not m:
            return None

        raw = m.group(1).strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                return parsed.strip()
        except Exception:
            pass

        if len(raw) >= 2 and raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]

        return raw.strip()

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

        if action == "open":
            return self.action_map(action, name=args.get("text", ""))

        if action == "key":
            return self.action_map(action, key_str=args.get("text", ""))

        clean_args = {k: v for k, v in args.items() if k != "action"}
        return self.action_map(action, **clean_args)

    def _quote(self, value: Any) -> str:
        return json.dumps("" if value is None else str(value), ensure_ascii=False)

    def action_map(self, action, **kwargs):
        if action == "click" or action == "left_click":
            x, y = kwargs["x"], kwargs["y"]
            return f'do(action="Tap", element=[{x}, {y}])'

        elif action == "long_press":
            x, y = kwargs["x"], kwargs["y"]
            return f'do(action="Long Press", element=[{x}, {y}])'

        elif action == "swipe":
            fx, fy = kwargs["from_x"], kwargs["from_y"]
            tx, ty = kwargs["to_x"], kwargs["to_y"]
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

        elif action == "terminate":
            status = kwargs["status"]
            return f'finish(message={self._quote(status)})'

        elif action == "answer":
            content = kwargs["content"]
            return f'finish(message={self._quote(content)})'

        else:
            raise ValueError(f"Unsupported action: {action}")
