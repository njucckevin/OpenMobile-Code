import base64
import copy
import re


def reverse_map_call_to_do(call_str: str) -> str:
    """
    将底层调用字符串反向映射成 do(...) 调用格式，
    以匹配 PixelLevelExecutor.current_return 里的 action/kwargs 结构。
    支持单/双引号。
    """
    call_str = call_str.strip()

    m = re.match(r'^click\(\s*x=([^,]+),\s*y=([^)]+)\)$', call_str)
    if m:
        x, y = m.group(1).strip(), m.group(2).strip()
        return f'do(action="Tap", element=[{x}, {y}])'

    m = re.match(
        r'^long_press\(\s*x=([^,]+),\s*y=([^,)]*)\s*(?:,\s*duration=([^)]+))?\s*\)$',
        call_str
    )
    if m:
        x = m.group(1).strip()
        y = m.group(2).strip()
        d = m.group(3).strip() if m.group(3) else None
        if d:
            return f'do(action="Long Press", element=[{x}, {y}], duration={d})'
        else:
            return f'do(action="Long Press", element=[{x}, {y}])'

    m = re.match(
        r'^swipe\(\s*from_coord=\s*([\(\[])\s*([^,]+)\s*,\s*([^,\)\]]+)\s*[\)\]]\s*,\s*'
        r'to_coord=\s*([\(\[])\s*([^,]+)\s*,\s*([^,\)\]]+)\s*[\)\]]\s*,\s*'
        r'direction=(["\'])(.+?)\7\s*\)$',
        call_str
    )
    if m:
        x1, y1 = m.group(2).strip(), m.group(3).strip()
        x2, y2 = m.group(5).strip(), m.group(6).strip()
        direction = m.group(8).strip()
        return (
            f'do(action="Swipe Precise", '
            f'start=[{x1}, {y1}], end=[{x2}, {y2}], direction="{direction}")'
        )

    m = re.match(
        r'^swipe\(\s*from_coord=\s*([\(\[])\s*([^,]+)\s*,\s*([^,\)\]]+)\s*[\)\]]\s*,\s*'
        r'to_coord=\s*([\(\[])\s*([^,]+)\s*,\s*([^,\)\]]+)\s*[\)\]]\s*\)$',
        call_str
    )
    if m:
        x1, y1, x2, y2 = m.group(2).strip(), m.group(3).strip(), m.group(5).strip(), m.group(6).strip()
        return (
            f'do(action="Swipe Precise", '
            f'start=[{x1}, {y1}], end=[{x2}, {y2}])'
        )

    m = re.match(
        r'^swipe\(\s*direction=(["\'])(.+?)\1\s*,\s*amount=([^)]+)\)$',
        call_str
    )
    if m:
        direction, amt = m.group(2), m.group(3).strip()
        return f'do(action="Swipe", direction="{direction}", dist={amt})'

    m = re.match(r'^write\(\s*message=(["\'])(.+?)\1\)$', call_str)
    if m:
        text = m.group(2)
        return f'do(action="Type", text="{text}")'

    m = re.match(r'^open_app\(\s*app_name=(["\'])(.+?)\1\)$', call_str)
    if m:
        pkg = m.group(2)
        return f'do(action="Launch", package="{pkg}")'

    if call_str == "navigate_home()":
        return 'do(action="Home")'

    if call_str == "navigate_back()":
        return 'do(action="Back")'

    if call_str == "enter()":
        return 'do(action="Enter")'

    m = re.match(r'^wait\(\s*seconds=([^)]+)\)$', call_str)
    if m:
        secs = m.group(1).strip()
        return f'do(action="Wait", seconds={secs})'

    if call_str == "call_user()":
        return 'do(action="Call_API")'

    m = re.match(r'^response\(\s*answer=(["\'])(.+?)\1\)$', call_str)
    if m:
        ans = m.group(2)
        return f'do(message="{ans}")'
    
    m = re.match(r"^terminate\(\s*status=(['\"])(.+?)\1\s*\)$", call_str)
    if m:
        return 'do(action="finish")'

    m = re.match(
        r'^terminate\(\s*status=(["\'])(?:.*?)\1\s*,\s*info=(["\'])(.+?)\2\)$',
        call_str
    )
    if m:
        msg = m.group(3)
        return f'do(action="finish", message="{msg}")'

    m = re.match(r'^response\(\s*answer=(["\'])(.+?)\1\)$', call_str)
    if m:
        ans = m.group(2)
        return f'do(message="{ans}")'



def parse_sections(text: str):
    pattern = re.compile(r'<(think|operation|action)>(.*?)</\1>', re.DOTALL)
    return {m.group(1): m.group(2).strip() for m in pattern.finditer(text)}

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_url(messages, throw_details=False, keep_path=False):
    new_messages = copy.deepcopy(messages)
    for message in new_messages:
        if message["role"] == "user":
            for content in message["content"]:
                if isinstance(content, str):
                    continue
                if content["type"] == "image_url":
                    image_url = content["image_url"]["url"]
                    image_url_parts = image_url.split(";base64,")
                    if not keep_path:
                        content["image_url"]["url"] = image_url_parts[0] + ";base64," + image_url_parts[1]
                    else:
                        content["image_url"]["url"] = f"file://{image_url_parts[1]}"
                    if throw_details:
                        content["image_url"].pop("detail", None)
    return new_messages
import math
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar
