# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Multimodal Autonomous Agent for Android (M3A)."""
import os
import re
import io
import time
import random

import ast
import json
from typing import Any
import numpy as np
from PIL import Image
from openai import OpenAI
from android_world.agents.PROMPT import *
import re

from openai import OpenAI
import time
import numpy as np
import requests
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from android_world.env import interface
from android_world.env import json_action
from android_world.agents import base_agent
import cv2
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.agents.utils import *
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils

# try:
#     from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (  # type: ignore
#         smart_resize,
#     )
# except Exception:  # pragma: no cover
#     smart_resize = None  # type: ignore[assignment]

def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """
    Rescale height/width to fit within [min_pixels, max_pixels] while keeping
    aspect ratio and ensuring dimensions are divisible by `factor`.
    
    This is a local implementation equivalent to:
    transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize
    
    Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
    """
    import math
    
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

# Utils for Visual Grounding


def _action_selection_prompt_locate(
    goal: str,
    history: list[str],
    ui_elements: str,
    additional_guidelines: list[str] | None = None,
) -> str:
    """Generate the prompt for the action selection.

    Args:
      goal: The current goal.
      history: Summaries for previous steps.
      ui_elements: A list of descriptions for the UI elements.
      additional_guidelines: Task specific guidelines.

    Returns:
      The text prompt for action selection that will be sent to gpt4v.
    """
    if history:
        history = "\n".join(history)
    else:
        history = "You just started, no action has been performed yet."

    extra_guidelines = ""
    if additional_guidelines:
        extra_guidelines = "For The Current Task:\n"
        for guideline in additional_guidelines:
            extra_guidelines += f"- {guideline}\n"

    return ACTION_SELECTION_PROMPT_TEMPLATE_LOCATE.format(
        goal=goal,
        history=history,
        additional_guidelines=extra_guidelines,
    )


def _generate_ui_element_description(
    ui_element: representation_utils.UIElement, index: int
) -> str:
    """Generate a description for a given UI element with important information.

    Args:
      ui_element: UI elements for the current screen.
      index: The numeric index for the UI element.

    Returns:
      The description for the UI element.
    """
    element_description = f'UI element {index}: {{"index": {index}, }}'
    if ui_element.text:
        element_description += f'"text": "{ui_element.text}", '
    if ui_element.content_description:
        element_description += (
            f'"content_description": "{ui_element.content_description}", '
        )
    if ui_element.hint_text:
        element_description += f'"hint_text": "{ui_element.hint_text}", '
    if ui_element.tooltip:
        element_description += f'"tooltip": "{ui_element.tooltip}", '
    element_description += (
        f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
    )
    element_description += (
        '"is_long_clickable":'
        f' {"True" if ui_element.is_long_clickable else "False"}, '
    )
    element_description += (
        f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
    )
    if ui_element.is_scrollable:
        element_description += '"is_scrollable": True, '
    if ui_element.is_focusable:
        element_description += '"is_focusable": True, '
    element_description += (
        f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
    )
    element_description += (
        f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
    )
    return element_description[:-2] + "}"


def _generate_ui_elements_description_list(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
    """Generate concise information for a list of UIElement.

    Args:
      ui_elements: UI elements for the current screen.
      screen_width_height_px: The height and width of the screen in pixels.

    Returns:
      Concise information for each UIElement.
    """
    tree_info = ""
    for index, ui_element in enumerate(ui_elements):
        if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
            tree_info += _generate_ui_element_description(ui_element, index) + "\n"
    return tree_info


def _summarize_prompt(
    action: str,
    reason: str,
    goal: str,
    before_elements: str,
    after_elements: str,
) -> str:
    """Generate the prompt for the summarization step.

    Args:
      action: Action picked.
      reason: The reason to pick the action.
      goal: The overall goal.
      before_elements: Information for UI elements on the before screenshot.
      after_elements: Information for UI elements on the after screenshot.

    Returns:
      The text prompt for summarization that will be sent to gpt4v.
    """
    return SUMMARY_PROMPT_TEMPLATE.format(
        goal=goal,
        before_elements=before_elements,
        after_elements=after_elements,
        action=action,
        reason=reason,
    )


class InternVL(base_agent.EnvironmentInteractingAgent):
    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "M3A",
        wait_after_action_seconds: float = 2.0,
        model_address="http://127.0.0.1:8000/",
        model_api_key="EMPTY",
        model_name="",
    ):
        super().__init__(env, name)
        self.llm = llm
        self.history = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = wait_after_action_seconds
        self.model_client = OpenAI(base_url=f"{model_address}v1", api_key=model_api_key)
        self.step_his: str = ""
        self.turn_number: int = 0
        self.model_name = model_name
        self.last_action = None
        self.repeat_time = 0

    def step(self, instruction: str) -> base_agent.AgentInteractionResult:
        self.turn_number += 1
        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()
        height, width = screenshot.shape[:2]

        system_prompt = internvl2_5_mobile_planning_cot_v1
        user_prompt = android_user_prompt.format(
            instruction=instruction, actions=self.step_his
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": self._to_base64_png(screenshot)},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        completion = self.model_client.chat.completions.create(
            model=self.model_name, messages=messages, temperature=0
        )
        response = completion.choices[0].message.content
        print(response)
        print("=" * 50)
        # Extract operation and actions
        op_m = re.search(r"<operation>([\s\S]*?)</operation>", response)
        act_m = re.search(r"<action>([\s\S]*?)</action>", response)
        op_text = op_m.group(1).strip() if op_m else ""
        self.step_his += f"Step {self.turn_number}: {op_text}\n"

        if not act_m:
            return base_agent.AgentInteractionResult(
                True, {"summary": "No valid action returned."}
            )
        if self.last_action == act_m.group(1):
            # return base_agent.AgentInteractionResult(True, {'operation': op_text, 'response': response})
            self.repeat_time += 1
        else:
            self.repeat_time = 0
        self.last_action = act_m.group(1)
        # Execute each parsed action
        cmds = [l for l in act_m.group(1).splitlines() if l.strip()]
        print(cmds)
        for cmd in cmds:
            parsed = action_transform(cmd, width, height)
            print(parsed)
            if not parsed:
                continue
            try:
                act = json_action.JSONAction(**parsed)
                self.env.execute_action(act)
                time.sleep(self.wait_after_action_seconds)
            except Exception:
                # continue
                print("Failed to execute action:", parsed)
        if "terminate" in response or self.repeat_time == 3:
            return base_agent.AgentInteractionResult(
                True, {"operation": op_text, "response": response}
            )
        return base_agent.AgentInteractionResult(
            False, {"operation": op_text, "response": response}
        )

    def get_point_from_description(
        self,
        image: np.ndarray,
        description: str,
    ) -> tuple[int, int]:
        def format_openai_template(description: str, base64_image):
            return [
                {"role": "system", "content": android_system_prompt_grounding},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": description},
                    ],
                },
            ]

        img = Image.fromarray(image)

        new_width = 1080
        new_height = 2340
        width, height = img.size

        print(width, height)

        img_resized = img.resize((new_width, new_height))

        if img_resized.mode == "RGBA":
            img_resized = img_resized.convert("RGB")

        img_byte_arr = io.BytesIO()
        img_resized.save(img_byte_arr, format="JPEG")
        image_bytes = img_byte_arr.getvalue()

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        messages = format_openai_template(description, base64_image)

        completion = self.model_client.chat.completions.create(
            model=self.model_name, messages=messages, temperature=0
        )

        response_text = completion.choices[0].message.content
        print(response_text)
        x_ratio, y_ratio = action_coord(response_text)
        print(f"x_ratio: {x_ratio}, y_ratio: {y_ratio}")

        x_coord = round(x_ratio * width)
        y_coord = round(y_ratio * height)

        return (x_coord, y_coord)

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self.history = []
        self.step_his = ""
        self.turn_number = 0
        self.last_action = None
        self.repeat_time = 0

    def step_planner(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data = {
            "raw_screenshot": None,
            "before_screenshot_with_som": None,
            "before_ui_elements": [],
            "after_screenshot_with_som": None,
            "action_prompt": None,
            "action_output": None,
            "action_output_json": None,
            "action_reason": None,
            "action_raw_response": None,
            "summary_prompt": None,
            "summary": None,
            "summary_raw_response": None,
        }
        print("----------step " + str(len(self.history) + 1))

        state = self.get_post_transition_state()
        step_data["raw_screenshot"] = state.pixels.copy()
        before_screenshot = state.pixels.copy()
        step_data["before_screenshot_with_som"] = before_screenshot.copy()

        action_prompt = _action_selection_prompt_locate(
            goal,
            [
                "Step " + str(i + 1) + "- " + step_info["summary"]
                for i, step_info in enumerate(self.history)
            ],
            None,
            self.additional_guidelines,
        )
        step_data["action_prompt"] = action_prompt
        action_output, is_safe, raw_response = self.llm.predict_mm(
            action_prompt,
            [
                step_data["raw_screenshot"],
            ],
        )

        if is_safe == False:
            action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""

        if not raw_response:
            raise RuntimeError("Error calling LLM in action selection phase.")
        step_data["action_output"] = action_output
        step_data["action_raw_response"] = raw_response

        reason, action = m3a_utils.parse_reason_action_output(action_output)

        if (not reason) or (not action):
            print("Action prompt output is not in the correct format.")
            step_data["summary"] = (
                "Output for action selection is not in the correct format, so no"
                " action is performed."
            )
            self.history.append(step_data)

            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )
        self.step_his += f"Step {self.turn_number}: {action}\n"
        print("Action: " + action)
        print("Reason: " + reason)
        step_data["action_reason"] = reason
        import traceback

        try:
            converted_action = json_action.JSONAction(
                **agent_utils.extract_json(action),
            )
            step_data["action_output_json"] = converted_action

            if converted_action.element:
                converted_action.x, converted_action.y = (
                    self.get_point_from_description(
                        step_data["raw_screenshot"], converted_action.element
                    )
                )

        except Exception as e:
            print("Failed to convert the output to a valid action.")
            print(traceback.print_exc())
            print(str(e))
            step_data["summary"] = (
                "Can not parse the output to a valid action. Please make sure to pick"
                " the action from the list with required parameters (if any) in the"
                " correct JSON format!"
            )
            self.history.append(step_data)

            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )
        if converted_action.action_type == "status":
            if converted_action.goal_status == "infeasible":
                print("Agent stopped since it thinks mission impossible.")
            step_data["summary"] = "Agent thinks the request has been completed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                True,
                step_data,
            )

        if converted_action.action_type == "answer":
            print("Agent answered with: " + converted_action.text)

        try:
            self.env.execute_action(converted_action)
        except Exception as e:
            print("Failed to execute action.")
            print(str(e))
            step_data["summary"] = (
                "Can not execute the action, make sure to select the action with"
                " the required parameters (if any) in the correct JSON format!"
            )
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        time.sleep(self.wait_after_action_seconds)

        state = self.env.get_state(wait_to_stabilize=False)

        after_screenshot = state.pixels.copy()

        if converted_action.x:
            m3a_utils.add_ui_element_dot(
                before_screenshot,
                target_element=(
                    [round(converted_action.x), round(converted_action.y)]
                    if converted_action.x
                    else None
                ),
            )

        step_data["before_screenshot_with_som"] = before_screenshot.copy()
        m3a_utils.add_screenshot_label(after_screenshot, "after")
        step_data["after_screenshot_with_som"] = after_screenshot.copy()

        summary_prompt = _summarize_prompt(
            action,
            reason,
            goal,
            None,
            None,
        )
        summary, is_safe, raw_response = self.llm.predict_mm(
            summary_prompt,
            [
                before_screenshot,
                after_screenshot,
            ],
        )

        if is_safe == False:
            summary = """Summary triggered LLM safety classifier."""

        if not raw_response:
            print(
                "Error calling LLM in summarization phase. This should not happen: "
                f"{summary}"
            )
            step_data["summary"] = (
                "Some error occurred calling LLM during summarization phase: %s"
                % summary
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        step_data["summary_prompt"] = summary_prompt
        step_data["summary"] = f"Action selected: {action}. {summary}"
        print("Summary: " + summary)
        step_data["summary_raw_response"] = raw_response

        self.history.append(step_data)
        return base_agent.AgentInteractionResult(
            False,
            step_data,
        )

    @staticmethod
    def _to_base64_png(image: np.ndarray) -> str:
        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        buf = BytesIO()
        PILImage.fromarray(image).save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


class QwenVL(base_agent.EnvironmentInteractingAgent):
    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "M3A",
        wait_after_action_seconds: float = 2.0,
        model_address="http://127.0.0.1:8000/",
        model_api_key="EMPTY",
        model_name="",
        mode="agent",
    ):
        super().__init__(env, name)
        self.llm = llm
        self.history = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = wait_after_action_seconds
        self.grounding_address = model_address
        self.model_client = OpenAI(base_url=f"{model_address}v1", api_key=model_api_key)
        self.step_his: str = ""
        self.turn_number: int = 0
        self.model_name = model_name
        self.last_action = None
        self.repeat_time = 0
        self.mode = mode  # 'agent' or 'grounder'

    def step(self, instruction: str) -> base_agent.AgentInteractionResult:
        if self.mode == "grounder":
            return self.step_planner(instruction)
        else:
            return self.step_agent(instruction)

    def step_agent(self, instruction: str) -> base_agent.AgentInteractionResult:
        self.turn_number += 1
        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()
        screenshot = screenshot[:, :, ::-1]
        if self.save_dir is not None:
            screenshot_path = os.path.join(
                self.save_dir, f"screenshot_{self.turn_number}.png"
            )
            cv2.imwrite(screenshot_path, screenshot)
            print(f"Screenshot saved to {screenshot_path}")
        height, width = screenshot.shape[:2]

        # system_prompt = internvl2_5_mobile_planning_cot_v1
        system_prompt = android_system_prompt_navigation
        user_prompt = android_user_prompt.format(
            instruction=instruction, actions=self.step_his
        )

        headers = {"Content-Type": "application/json"}
        url = f"{self.grounding_address}v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self._to_base64_png(screenshot)},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        }
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response = response.json()["choices"][0]["message"]["content"]
        print(response)
        print("=" * 50)
        op_m = re.search(r"<operation>([\s\S]*?)</operation>", response)
        act_m = re.search(r"<action>([\s\S]*?)</action>", response)
        op_text = op_m.group(1).strip() if op_m else ""
        self.step_his += f"Step {self.turn_number}: {op_text}\n"

        if not act_m:
            return base_agent.AgentInteractionResult(
                True, {"summary": "No valid action returned."}
            )
        if self.last_action == act_m.group(1):
            self.repeat_time += 1
        else:
            self.repeat_time = 0
        self.last_action = act_m.group(1)
        cmds = [l for l in act_m.group(1).splitlines() if l.strip()]
        print(cmds)
        for cmd in cmds:
            parsed = qwen_action_transform(
                cmd,
                width,
                height,
                smart_resize_option=True,
                min_pixels=3136,
                max_pixels=2109744,
            )
            print(parsed)
            if not parsed:
                continue
            try:
                act = json_action.JSONAction(**parsed)
                self.env.execute_action(act)
                time.sleep(self.wait_after_action_seconds)
            except Exception:
                print("Failed to execute action:", parsed)
        if "terminate" in response or self.repeat_time == 3:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "operation": op_text,
                    "response": response,
                    "step_history": self.step_his,
                },
            )
        return base_agent.AgentInteractionResult(
            False,
            {"operation": op_text, "response": response, "step_history": self.step_his},
        )

    def get_point_from_description(
        self,
        image: np.ndarray,
        description: str,
    ) -> tuple[int, int]:

        def format_openai_template(description: str, base64_image):
            return [
                {"role": "system", "content": android_system_prompt_grounding},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": description,
                        },
                    ],
                },
            ]

        img = Image.fromarray(image)

        new_width = 1080
        new_height = 2340
        width, height = img.size

        print(width, height)

        img_resized = img.resize((new_width, new_height))

        if img_resized.mode == "RGBA":
            img_resized = img_resized.convert("RGB")

        img_byte_arr = io.BytesIO()
        img_resized.save(img_byte_arr, format="JPEG")
        image_bytes = img_byte_arr.getvalue()

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        messages = format_openai_template(description, base64_image)

        headers = {"Content-Type": "application/json"}
        url = f"{self.grounding_address}v1/chat/completions"
        payload = {"model": self.model_name, "messages": messages}
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_text = response.json()["choices"][0]["message"]["content"]
        print(response_text)
        x_ratio, y_ratio = action_coord(response_text)
        print(f"x_ratio: {x_ratio}, y_ratio: {y_ratio}")
        resized_height, resized_width = smart_resize(
            height, width, min_pixels=3136, max_pixels=2109744
        )
        x_coord = round(x_ratio / resized_width * width)
        y_coord = round(y_ratio / resized_height * height)

        return (x_coord, y_coord)

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self.history = []
        self.step_his = ""
        self.turn_number = 0
        self.last_action = None
        self.repeat_time = 0

    def step_planner(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data = {
            "raw_screenshot": None,
            "before_screenshot_with_som": None,
            "before_ui_elements": [],
            "after_screenshot_with_som": None,
            "action_prompt": None,
            "action_output": None,
            "action_output_json": None,
            "action_reason": None,
            "action_raw_response": None,
            "summary_prompt": None,
            "summary": None,
            "summary_raw_response": None,
            "response": None,
            "step_history": None,
        }
        print("----------step " + str(len(self.history) + 1))

        state = self.get_post_transition_state()
        step_data["raw_screenshot"] = state.pixels.copy()
        before_screenshot = state.pixels.copy()
        step_data["before_screenshot_with_som"] = before_screenshot.copy()

        action_prompt = _action_selection_prompt_locate(
            goal,
            [
                "Step " + str(i + 1) + "- " + step_info["summary"]
                for i, step_info in enumerate(self.history)
            ],
            None,
            self.additional_guidelines,
        )
        step_data["action_prompt"] = action_prompt
        action_output, is_safe, raw_response = self.llm.predict_mm(
            action_prompt,
            [
                step_data["raw_screenshot"],
            ],
        )

        if is_safe == False:
            action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""

        if not raw_response:
            raise RuntimeError("Error calling LLM in action selection phase.")
        step_data["action_output"] = action_output
        step_data["action_raw_response"] = raw_response

        reason, action = m3a_utils.parse_reason_action_output(action_output)
        if (not reason) or (not action):
            print("Action prompt output is not in the correct format.")
            step_data["summary"] = (
                "Output for action selection is not in the correct format, so no"
                " action is performed."
            )
            self.history.append(step_data)

            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )
        self.step_his += f"Step {self.turn_number}: {action}\n"
        step_data["step_history"] = self.step_his
        step_data["response"] = action_output
        print("Action: " + action)
        print("Reason: " + reason)
        step_data["action_reason"] = reason
        import traceback

        try:
            converted_action = json_action.JSONAction(
                **agent_utils.extract_json(action),
            )
            step_data["action_output_json"] = converted_action

            if converted_action.element:
                converted_action.x, converted_action.y = (
                    self.get_point_from_description(
                        step_data["raw_screenshot"], converted_action.element
                    )
                )

        except Exception as e:
            print("Failed to convert the output to a valid action.")
            print(traceback.print_exc())
            print(str(e))
            step_data["summary"] = (
                "Can not parse the output to a valid action. Please make sure to pick"
                " the action from the list with required parameters (if any) in the"
                " correct JSON format!"
            )
            self.history.append(step_data)

            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )
        if converted_action.action_type == "status":
            if converted_action.goal_status == "infeasible":
                print("Agent stopped since it thinks mission impossible.")
            step_data["summary"] = "Agent thinks the request has been completed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                True,
                step_data,
            )

        if converted_action.action_type == "answer":
            print("Agent answered with: " + converted_action.text)

        try:
            self.env.execute_action(converted_action)
        except Exception as e:
            print("Failed to execute action.")
            print(str(e))
            step_data["summary"] = (
                "Can not execute the action, make sure to select the action with"
                " the required parameters (if any) in the correct JSON format!"
            )
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        time.sleep(self.wait_after_action_seconds)

        state = self.env.get_state(wait_to_stabilize=False)

        after_screenshot = state.pixels.copy()
        if converted_action.x:
            m3a_utils.add_ui_element_dot(
                before_screenshot,
                target_element=(
                    [round(converted_action.x), round(converted_action.y)]
                    if converted_action.x
                    else None
                ),
            )

        step_data["before_screenshot_with_som"] = before_screenshot.copy()
        m3a_utils.add_screenshot_label(after_screenshot, "after")
        step_data["after_screenshot_with_som"] = after_screenshot.copy()

        summary_prompt = _summarize_prompt(
            action,
            reason,
            goal,
            None,
            None,
        )
        summary, is_safe, raw_response = self.llm.predict_mm(
            summary_prompt,
            [
                before_screenshot,
                after_screenshot,
            ],
        )

        if is_safe == False:
            summary = """Summary triggered LLM safety classifier."""

        if not raw_response:
            print(
                "Error calling LLM in summarization phase. This should not happen: "
                f"{summary}"
            )
            step_data["summary"] = (
                "Some error occurred calling LLM during summarization phase: %s"
                % summary
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        step_data["summary_prompt"] = summary_prompt
        step_data["summary"] = f"Action selected: {action}. {summary}"
        print("Summary: " + summary)
        step_data["summary_raw_response"] = raw_response

        self.history.append(step_data)
        return base_agent.AgentInteractionResult(
            False,
            step_data,
        )

    @staticmethod
    def _to_base64_png(image: np.ndarray) -> str:
        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        buf = BytesIO()
        PILImage.fromarray(image).save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _extract_action_text_qwen3vl(block: str) -> str:
    """Extracts the 'Action:' line from Qwen3VL text output for step history (does not affect execution)."""
    m = re.search(r"Action:\s*(.+?)(?:\n<tool_call>|$)", block, flags=re.S)
    if not m:
        return ""
    text = m.group(1).strip()
    # Some models wrap Action: "..." with quotes.
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text.replace("\n", " ")


def _extract_action_text_qwen3vl_gemini(block: str) -> str:
    """Extract the <thinking>...</thinking> content (for Gemini-style outputs)."""
    # # If the model outputs an explicit "Action:" line, reuse the original parser.
    # if re.search(r"\baction\s*:", block, flags=re.I):
    #     return _extract_action_text_qwen3vl(block)
    m = re.search(r"<thinking>\s*([\s\S]*?)\s*</thinking>", block)
    if m:
        return m.group(1).strip()
    # Fallback: extract prefix before tool_call.
    m2 = re.search(r"^\s*([\s\S]*?)(?=<tool_call>|$)", block)
    if not m2:
        return ""
    return m2.group(1).strip()

def _extract_conclusion_text_qwen3vl_gemini(block: str) -> str:
    """Extract the <conclusion>...</conclusion> content (for Gemini-style outputs)."""
    m = re.search(r"<conclusion>\s*([\s\S]*?)\s*</conclusion>", block)
    return m.group(1).strip() if m else ""


def _parse_tool_call_json(block: str) -> dict[str, Any] | None:
    """Parse JSON inside <tool_call>...</tool_call>."""
    m = re.search(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", block)
    if not m:
        return None
    payload = m.group(1).strip()
    try:
        return json.loads(payload)
    except Exception:
        return None


def _parse_tool_call_xml(block: str) -> dict[str, Any] | None:
    """Parse Qwen3.5 XML function calls into the existing tool-call dict shape."""
    m = re.search(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", block or "")
    if not m:
        return None
    payload = m.group(1).strip()
    func = re.search(r"<function=([^>\s]+)>\s*([\s\S]*?)\s*</function>", payload)
    if not func:
        return None

    arguments: dict[str, Any] = {}
    for param_name, raw_value in re.findall(
        r"<parameter=([^>\s]+)>\s*([\s\S]*?)\s*</parameter>", func.group(2)
    ):
        value = raw_value.strip()
        try:
            arguments[param_name] = json.loads(value)
        except Exception:
            arguments[param_name] = value
    return {"name": func.group(1).strip(), "arguments": arguments}


def _parse_native_tool_call(message: Any) -> dict[str, Any] | None:
    """Convert OpenAI/vLLM native tool_calls into the existing tool-call dict shape."""
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return None
    call = tool_calls[0]
    function = getattr(call, "function", None)
    if function is None:
        return None
    raw_args = getattr(function, "arguments", "{}") or "{}"
    try:
        args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
    except Exception:
        return None
    return {"name": getattr(function, "name", "mobile_use"), "arguments": args}


def _extract_thinking_text(block: str) -> str:
    m = re.search(r"<thinking>\s*([\s\S]*?)\s*</thinking>", block or "", flags=re.I)
    return m.group(1).strip() if m else ""


def _extract_conclusion_text(block: str) -> str:
    m = re.search(r"<conclusion>\s*([\s\S]*?)\s*</conclusion>", block or "", flags=re.I)
    return m.group(1).strip() if m else ""


def _extract_history_summary(block: str) -> str:
    """Extract a compact summary string for step history across multiple output styles."""
    conclusion = _extract_conclusion_text(block)
    if conclusion:
        return conclusion.replace("\n", " ")
    thinking = _extract_thinking_text(block)
    if thinking:
        return thinking.replace("\n", " ")
    action_line = _extract_action_text_qwen3vl(block)
    if action_line:
        return action_line
    m = re.search(r"^\s*([\s\S]*?)(?=<tool_call>|$)", block or "")
    return m.group(1).strip().replace("\n", " ") if m else ""


def _scale_coord_resized_to_1000(coord: Any, resized_w: int, resized_h: int) -> list[int]:
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return [0, 0]
    x = float(coord[0])
    y = float(coord[1])
    if resized_w <= 0 or resized_h <= 0:
        return [0, 0]
    xn = int(round(x / float(resized_w) * 1000.0))
    yn = int(round(y / float(resized_h) * 1000.0))
    xn = max(0, min(1000, xn))
    yn = max(0, min(1000, yn))
    return [xn, yn]


def _convert_qwen25_tool_call_to_qwen3(
    tool_call: dict[str, Any],
    resized_w: int,
    resized_h: int,
) -> dict[str, Any]:
    """Convert Qwen2.5VL resized-pixel coords to Qwen3-style 0~1000 coords for logging."""
    if not isinstance(tool_call, dict):
        return {"name": "mobile_use", "arguments": {"action": "wait"}}
    args = tool_call.get("arguments", {}) if isinstance(tool_call.get("arguments"), dict) else {}
    out_args = dict(args)
    action = out_args.get("action")
    if action in {"click", "left_click", "long_press", "swipe"}:
        if "coordinate" in out_args:
            out_args["coordinate"] = _scale_coord_resized_to_1000(
                out_args.get("coordinate"),
                resized_w,
                resized_h,
            )
        if action == "swipe" and "coordinate2" in out_args:
            out_args["coordinate2"] = _scale_coord_resized_to_1000(
                out_args.get("coordinate2"),
                resized_w,
                resized_h,
            )
    return {
        "name": tool_call.get("name", "mobile_use"),
        "arguments": out_args,
    }


def _replace_tool_call_block(response: str, tool_call: dict[str, Any]) -> str:
    tool_call_text = "<tool_call>\n" + json.dumps(tool_call, ensure_ascii=False) + "\n</tool_call>"
    if re.search(r"<tool_call>\s*[\s\S]*?\s*</tool_call>", response or ""):
        return re.sub(
            r"<tool_call>\s*[\s\S]*?\s*</tool_call>",
            tool_call_text,
            response,
            count=1,
        )
    base = (response or "").strip()
    if base:
        return base + "\n" + tool_call_text
    return tool_call_text


def _pick_min_area_bbox_for_point(
    ui_elements: list[representation_utils.UIElement],
    x: int,
    y: int,
) -> dict[str, int] | None:
    best = None
    best_area = None
    for el in ui_elements or []:
        bbox = getattr(el, "bbox_pixels", None)
        if bbox is None:
            continue
        try:
            x_min = int(getattr(bbox, "x_min"))
            x_max = int(getattr(bbox, "x_max"))
            y_min = int(getattr(bbox, "y_min"))
            y_max = int(getattr(bbox, "y_max"))
        except Exception:
            continue
        if x_max <= x_min or y_max <= y_min:
            continue
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue
        area = (x_max - x_min) * (y_max - y_min)
        if best_area is None or area < best_area:
            best_area = area
            best = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    return best


def _point_in_bbox(x: int, y: int, bbox: dict[str, int]) -> bool:
    return (
        bbox["x_min"] <= x <= bbox["x_max"]
        and bbox["y_min"] <= y <= bbox["y_max"]
    )


def _text_contains_either(a: str, b: str) -> bool:
    aa = (a or "").strip().lower()
    bb = (b or "").strip().lower()
    if not aa and not bb:
        return True
    if not aa or not bb:
        return False
    return aa in bb or bb in aa


def _ui_element_to_metadata_dict(element: representation_utils.UIElement) -> dict[str, Any]:
    """Convert UIElement to a JSON-serializable dict (keep bbox for post-processing/RL)."""
    bbox = getattr(element, "bbox_pixels", None)
    bbox_dict = (
        {
            "x_min": getattr(bbox, "x_min", None),
            "x_max": getattr(bbox, "x_max", None),
            "y_min": getattr(bbox, "y_min", None),
            "y_max": getattr(bbox, "y_max", None),
        }
        if bbox is not None
        else None
    )
    return {
        "bbox_pixels": bbox_dict,
        "resource_id": getattr(element, "resource_id", None),
        "resource_name": getattr(element, "resource_name", None),
        "text": getattr(element, "text", None),
        "content_description": getattr(element, "content_description", None),
        "class_name": getattr(element, "class_name", None),
        "hint_text": getattr(element, "hint_text", None),
        "package_name": getattr(element, "package_name", None),
        "is_checkable": getattr(element, "is_checkable", None),
        "is_enabled": getattr(element, "is_enabled", None),
        "is_visible": getattr(element, "is_visible", None),
        "is_clickable": getattr(element, "is_clickable", None),
        "is_editable": getattr(element, "is_editable", None),
        "is_focused": getattr(element, "is_focused", None),
        "is_focusable": getattr(element, "is_focusable", None),
        "is_long_clickable": getattr(element, "is_long_clickable", None),
        "is_scrollable": getattr(element, "is_scrollable", None),
        "is_selected": getattr(element, "is_selected", None),
    }


MONITOR_SYSTEM_PROMPT = """You are a supervisor monitoring an Android GUI agent trajectory.

Your job is to decide whether the weak acting agent should continue acting on the current step, or whether a stronger agent should intervene starting from the current step.

You will be given:
- the user instruction
- the text action history so far
- the most recent screenshots in chronological order, where the last screenshot is the current screen after the most recent action was executed

Judge based on the overall execution process rather than whether there exists a different possible path. Alternative but still plausible paths are allowed. However, if the execution appears to have clearly drifted away from the task, become confused, or likely made a significant mistake, you should intervene immediately.

Return ONLY valid JSON in this exact schema:
{
  "should_intervene": true or false,
  "reason": "brief explanation of why the current execution is or is not off track",
  "summary": "very short summary"
}
"""


def _chat_with_retry(
    client: OpenAI,
    model_name: str,
    messages: list[dict[str, Any]],
    retry_times: int = 5,
    retry_sleep_s: float = 10.0,
    retry_name: str = "model",
    request_timeout_s: float = 60.0,
) -> str:
    response = ""
    for _ in range(retry_times):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                timeout=request_timeout_s,
            )
        except Exception as e:
            print(
                f"{retry_name} request failed or timed out after "
                f"{int(request_timeout_s)}s: {e}"
            )
            response = ""
            continue
        try:
            response = completion.choices[0].message.content or ""
        except Exception:
            response = ""
        if response.strip():
            break
        print(
            f"{retry_name} empty response, sleep {int(retry_sleep_s)} seconds and retry"
        )
        time.sleep(retry_sleep_s)
    return response


def _parse_monitor_output(response: str) -> dict[str, Any]:
    raw = (response or "").strip()
    payload = raw
    try:
        return json.loads(payload)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        payload = match.group(0)
        try:
            return json.loads(payload)
        except Exception:
            pass
    return {
        "should_intervene": False,
        "reason": f"Failed to parse monitor output: {raw[:500]}",
        "summary": "parse_failed",
    }


class Qwen3VL(base_agent.EnvironmentInteractingAgent):
    """Android GUI Agent based on Qwen3VL tool-call output (for AndroidWorld eval).

    - Input: Screenshot + instruction + history
    - Output: <tool_call>{...}</tool_call>
    - Execution: Map to JSONAction by qwen3vl_action_transform(...)
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "Qwen3VL",
        wait_after_action_seconds: float = 2.0,
        model_base_url: str = "http://127.0.0.1:8000/v1",
        model_api_key: str = "EMPTY",
        model_name: str = "",
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(env, name)
        self.llm = llm
        self.wait_after_action_seconds = wait_after_action_seconds
        self.model_name = model_name
        self.client = OpenAI(
            api_key=model_api_key,
            base_url=model_base_url,
            default_headers=extra_headers,
        )

        # Used for self-deployed model (Not Used)
        self.model_base_url = model_base_url
        
        self.step_his: str = ""
        self.turn_number: int = 0

        # Provide multiple most recent screenshots to the model (hard-coded; user may adjust).
        self.last_N = 1
        self._recent_screenshots = deque(maxlen=self.last_N)

        # Used to detect repeated actions (avoid infinite loops)
        self.last_action: str | None = None
        self.repeat_time: int = 0

        # Per-step ui_elements metadata aligned with screenshot_step{step}.png.
        self._ui_elements_history: list[dict[str, Any]] = []

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self.step_his = ""
        self.turn_number = 0
        self._recent_screenshots.clear()
        self.last_action = None
        self.repeat_time = 0
        self._ui_elements_history = []

    @staticmethod
    def _to_base64_png(image: np.ndarray) -> str:
        import base64
        from io import BytesIO
        from PIL import Image as PILImage
        buf = BytesIO()
        PILImage.fromarray(image).save(buf, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def step(self, instruction: str) -> base_agent.AgentInteractionResult:
        self.turn_number += 1

        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()  # RGB format from Android

        # Save per-step ui_elements for post-processing (e.g., click_point -> bbox).
        # We write metadata.json every step (overwrite) to avoid needing an explicit "episode end" hook.
        if self.save_dir is not None:
            try:
                step_idx = self.turn_number - 1
                self._ui_elements_history.append({
                    "step": step_idx,
                    "logical_screen_size": list(self.env.logical_screen_size),
                    "ui_elements": [_ui_element_to_metadata_dict(e) for e in state.ui_elements],
                })
                meta = {"goal": instruction, "steps": self._ui_elements_history}
                with open(os.path.join(self.save_dir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Failed to save ui_elements metadata: {e}")
        
        # Save screenshot
        if self.save_dir is not None:
            try:
                screenshot_path = os.path.join(self.save_dir, f"screenshot_step{self.turn_number - 1}.png")
                Image.fromarray(screenshot).save(screenshot_path)
            except Exception as e:
                print(f"Failed to save screenshot: {e}")
        
        self._recent_screenshots.append(screenshot)
        height, width = screenshot.shape[:2]

        # Use Gemini3Pro prompt format when model name contains "gemini".
        if "gemini" in (self.model_name or "").lower():
            system_prompt = GEMINI3PRO_SYSTEM_PROMPT
            user_prompt = GEMINI3PRO_USER_PROMPT.format(
                instruction=instruction, history=self.step_his
            )
        else:
            system_prompt = QWEN3VL_SYSTEM_PROMPT
            user_prompt = QWEN3VL_USER_PROMPT.format(
                instruction=instruction, history=self.step_his
            )
        print(user_prompt)

        user_content = [{"type": "text", "text": user_prompt}]
        for img in list(self._recent_screenshots):
            user_content.append(
                {"type": "image_url", "image_url": {"url": self._to_base64_png(img)}}
            )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        response = ""
        completion = None
        for _ in range(5):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    timeout=60.0,
                )
            except Exception as e:
                print(f"Qwen3VL request failed or timed out after 60s: {e}")
                response = ""
                continue
            # print(completion)
            try:
                response = completion.choices[0].message.content or ""
            except Exception:
                response = ""
            if response.strip():
                break
            print("sleep 10 seconds and retry")
            time.sleep(10)

        print(response)
        print("=" * 50)

        tool_call = _parse_tool_call_json(response)
        if not tool_call:
            return base_agent.AgentInteractionResult(
                True, {"summary": "No <tool_call> JSON found in model output.", "response": response}
            )

        if "gemini" in (self.model_name or "").lower():
            conclusion_text = _extract_conclusion_text_qwen3vl_gemini(response)
            thinking_text = _extract_action_text_qwen3vl_gemini(response)
            # Align with Qwen2.5VL history logic: prefer conclusion, fallback to thinking.
            op_text = conclusion_text if conclusion_text else thinking_text
        else:
            op_text = _extract_action_text_qwen3vl(response)
        self.step_his += f"Step {self.turn_number}: {op_text}; "

        # Compatible: tool_call may look like {"name":"mobile_use","arguments":{...}}
        args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}
        action_name = args.get("action", "")
        try:
            parsed = qwen3vl_action_transform(action_name, args, width, height)
            print(parsed)
        except Exception as e:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": f"Failed to transform tool-call into action: {e}",
                    "response": response,
                    "tool_call": tool_call,
                },
            )

        # If model outputs an answer, persist it and stop immediately.
        if parsed.get("action_type") == "answer":
            try:
                act = json_action.JSONAction(**parsed)
                self.env.execute_action(act)
            except Exception:
                print("Failed to execute answer action:", parsed)
            return base_agent.AgentInteractionResult(
                True, {"response": response, "step_history": self.step_his, "parsed": parsed}
            )

        # Record last_action + repeat_time (previous code had these fields but not working)
        # Here, use the tool-call's arguments as the "action signature", which is more robust than checking 'terminate' in a string.
        try:
            action_sig = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except Exception:
            action_sig = str(args)
        if self.last_action == action_sig:
            self.repeat_time += 1
        else:
            self.repeat_time = 0
        self.last_action = action_sig

        try:
            act = json_action.JSONAction(**parsed)
            self.env.execute_action(act)
            time.sleep(self.wait_after_action_seconds)
        except Exception:
            # continue
            print("Failed to execute action:", parsed)

        if parsed.get("action_type") == "status":
            return base_agent.AgentInteractionResult(
                True, {"response": response, "step_history": self.step_his, "parsed": parsed}
            )

        # If repeated actions reach the threshold: terminate immediately to avoid deadlock in evaluation
        if self.repeat_time >= 10:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": "Terminated due to repeated identical actions.",
                    "response": response,
                    "step_history": self.step_his,
                    "parsed": parsed,
                    "repeat_time": self.repeat_time,
                },
            )

        return base_agent.AgentInteractionResult(
            False, {"response": response, "step_history": self.step_his, "parsed": parsed}
        )


class Qwen35VL(Qwen3VL):
    """Qwen3.5 GUI agent with switchable native/XML tool-call handling."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "Qwen35VL",
        wait_after_action_seconds: float = 2.0,
        model_base_url: str = "http://127.0.0.1:8000/v1",
        model_api_key: str = "EMPTY",
        model_name: str = "",
        extra_headers: dict[str, str] | None = None,
        qwen35_tool_call_mode: str = "xml",
    ):
        super().__init__(
            env=env,
            llm=llm,
            name=name,
            wait_after_action_seconds=wait_after_action_seconds,
            model_base_url=model_base_url,
            model_api_key=model_api_key,
            model_name=model_name,
            extra_headers=extra_headers,
        )
        if qwen35_tool_call_mode not in {"native", "xml"}:
            raise ValueError(
                "qwen35_tool_call_mode must be either 'native' or 'xml', "
                f"got {qwen35_tool_call_mode!r}"
            )
        self.qwen35_tool_call_mode = qwen35_tool_call_mode

    def step(self, instruction: str) -> base_agent.AgentInteractionResult:
        self.turn_number += 1

        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()  # RGB format from Android

        # Save per-step ui_elements for post-processing (e.g., click_point -> bbox).
        # We write metadata.json every step (overwrite) to avoid needing an explicit "episode end" hook.
        if self.save_dir is not None:
            try:
                step_idx = self.turn_number - 1
                self._ui_elements_history.append({
                    "step": step_idx,
                    "logical_screen_size": list(self.env.logical_screen_size),
                    "ui_elements": [_ui_element_to_metadata_dict(e) for e in state.ui_elements],
                })
                meta = {"goal": instruction, "steps": self._ui_elements_history}
                with open(os.path.join(self.save_dir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Failed to save ui_elements metadata: {e}")

        # Save screenshot
        if self.save_dir is not None:
            try:
                screenshot_path = os.path.join(self.save_dir, f"screenshot_step{self.turn_number - 1}.png")
                Image.fromarray(screenshot).save(screenshot_path)
            except Exception as e:
                print(f"Failed to save screenshot: {e}")

        self._recent_screenshots.append(screenshot)
        height, width = screenshot.shape[:2]

        use_native_tools = self.qwen35_tool_call_mode == "native"
        if use_native_tools:
            system_prompt = QWEN35_NATIVE_SYSTEM_PROMPT
        else:
            system_prompt = QWEN35_SYSTEM_PROMPT
        user_prompt = QWEN35_USER_PROMPT.format(
            instruction=instruction, history=self.step_his
        )
        print(user_prompt)

        user_content = [{"type": "text", "text": user_prompt}]
        for img in list(self._recent_screenshots):
            user_content.append(
                {"type": "image_url", "image_url": {"url": self._to_base64_png(img)}}
            )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        response = ""
        completion = None
        completion_message = None
        response_content = ""

        if use_native_tools:
            for _ in range(5):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0,
                        timeout=60.0,
                        tools=QWEN35_TOOLS,
                        tool_choice="auto",
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    )
                except Exception as e:
                    print(f"Qwen35VL request failed or timed out after 60s: {e}")
                    response = ""
                    continue
                # print(completion)
                try:
                    completion_message = completion.choices[0].message
                    response_content = completion_message.content or ""
                    response = completion_message.model_dump_json(
                        indent=2, exclude_none=True
                    )
                except Exception:
                    response = ""
                    response_content = ""
                    completion_message = None
                if response.strip() or getattr(completion_message, "tool_calls", None):
                    break
                print("sleep 10 seconds and retry")
                time.sleep(10)
        else:
            for _ in range(5):
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0,
                        timeout=60.0,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False}
                        },
                    )
                except Exception as e:
                    print(f"Qwen35VL request failed or timed out after 60s: {e}")
                    response = ""
                    continue
                # print(completion)
                try:
                    response = completion.choices[0].message.content or ""
                    response_content = response
                except Exception:
                    response = ""
                    response_content = ""
                if response.strip():
                    break
                print("sleep 10 seconds and retry")
                time.sleep(10)

        print(response)
        print("=" * 50)

        if use_native_tools:
            tool_call = _parse_native_tool_call(completion_message)
            if not tool_call:
                print(f"No valid tool_call found in Qwen3.5 output: {response}")
                tool_call = _parse_tool_call_xml(response)
        else:
            tool_call = _parse_tool_call_xml(response)
        if not tool_call:
            return base_agent.AgentInteractionResult(
                True, {"summary": "No valid <tool_call> found in model output.", "response": response}
            )

        op_text = _extract_action_text_qwen3vl(response_content)
        self.step_his += f"Step {self.turn_number}: {op_text}; "

        # Compatible: tool_call may look like {"name":"mobile_use","arguments":{...}}
        args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}
        action_name = args.get("action", "")
        try:
            parsed = qwen3vl_action_transform(action_name, args, width, height)
            print(parsed)
        except Exception as e:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": f"Failed to transform tool-call into action: {e}",
                    "response": response,
                    "tool_call": tool_call,
                },
            )

        # If model outputs an answer, persist it and stop immediately.
        if parsed.get("action_type") == "answer":
            try:
                act = json_action.JSONAction(**parsed)
                self.env.execute_action(act)
            except Exception:
                print("Failed to execute answer action:", parsed)
            return base_agent.AgentInteractionResult(
                True, {"response": response, "step_history": self.step_his, "parsed": parsed}
            )

        # Record last_action + repeat_time (previous code had these fields but not working)
        # Here, use the tool-call's arguments as the "action signature", which is more robust than checking 'terminate' in a string.
        try:
            action_sig = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except Exception:
            action_sig = str(args)
        if self.last_action == action_sig:
            self.repeat_time += 1
        else:
            self.repeat_time = 0
        self.last_action = action_sig

        try:
            act = json_action.JSONAction(**parsed)
            self.env.execute_action(act)
            time.sleep(self.wait_after_action_seconds)
        except Exception:
            # continue
            print("Failed to execute action:", parsed)

        if parsed.get("action_type") == "status":
            return base_agent.AgentInteractionResult(
                True, {"response": response, "step_history": self.step_his, "parsed": parsed}
            )

        # If repeated actions reach the threshold: terminate immediately to avoid deadlock in evaluation
        if self.repeat_time >= 10:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": "Terminated due to repeated identical actions.",
                    "response": response,
                    "step_history": self.step_his,
                    "parsed": parsed,
                    "repeat_time": self.repeat_time,
                },
            )

        return base_agent.AgentInteractionResult(
            False, {"response": response, "step_history": self.step_his, "parsed": parsed}
        )


class Qwen3VL_Switching(base_agent.EnvironmentInteractingAgent):
    """Weak-first rollout with monitor-triggered intervention."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "Qwen3VL_Switching",
        wait_after_action_seconds: float = 2.0,
        model_base_url: str = "http://127.0.0.1:8000/v1",
        model_api_key: str = "EMPTY",
        model_name: str = "",
        extra_headers: dict[str, str] | None = None,
        weak_model_base_url: str = "http://127.0.0.1:8001/v1",
        weak_model_api_key: str = "EMPTY",
        weak_model_name: str = "",
        weak_min_pixels: int = 3136,
        weak_max_pixels: int = 12845056,
        monitor_num_recent_screens: int = 2,
        max_interventions: int = 2,
        strong_min_steps_after_intervention: int = 3,
    ):
        super().__init__(env, name)
        self.llm = llm
        self.wait_after_action_seconds = wait_after_action_seconds

        self.model_name = model_name
        self.strong_client = OpenAI(
            api_key=model_api_key,
            base_url=model_base_url,
            default_headers=extra_headers,
        )
        self.monitor_client = OpenAI(
            api_key=model_api_key,
            base_url=model_base_url,
            default_headers=extra_headers,
        )
        self.weak_client = OpenAI(
            api_key=weak_model_api_key,
            base_url=weak_model_base_url,
            default_headers=extra_headers,
        )

        self.step_his: str = ""
        self.turn_number: int = 0
        self.last_N = 1
        self._recent_screenshots = deque(maxlen=max(1, monitor_num_recent_screens))
        self.monitor_num_recent_screens = max(1, monitor_num_recent_screens)

        self.last_action: str | None = None
        self.repeat_time: int = 0

        self.current_policy: str = "weak"
        self.has_intervened: bool = False
        self.pending_intervention_feedback: str | None = None
        self.max_interventions = max(1, max_interventions)
        self.strong_min_steps_after_intervention = max(
            1, strong_min_steps_after_intervention
        )
        self.intervention_count: int = 0
        self.strong_steps_since_intervention: int = 0
        self.should_monitor_next_step: bool = False

        self.weak_model_name = weak_model_name
        self.weak_min_pixels = weak_min_pixels
        self.weak_max_pixels = weak_max_pixels

        self._ui_elements_history: list[dict[str, Any]] = []

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self.step_his = ""
        self.turn_number = 0
        self._recent_screenshots.clear()
        self.last_action = None
        self.repeat_time = 0
        self.current_policy = "weak"
        self.has_intervened = False
        self.pending_intervention_feedback = None
        self.intervention_count = 0
        self.strong_steps_since_intervention = 0
        self.should_monitor_next_step = False
        self._ui_elements_history = []

    @staticmethod
    def _to_base64_png(image: np.ndarray) -> str:
        import base64
        from io import BytesIO
        from PIL import Image as PILImage

        buf = BytesIO()
        PILImage.fromarray(image).save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def _save_step_artifacts(
        self,
        instruction: str,
        state: interface.State,
        screenshot: np.ndarray,
    ) -> None:
        if self.save_dir is not None:
            try:
                step_idx = self.turn_number - 1
                self._ui_elements_history.append(
                    {
                        "step": step_idx,
                        "logical_screen_size": list(self.env.logical_screen_size),
                        "ui_elements": [
                            _ui_element_to_metadata_dict(e) for e in state.ui_elements
                        ],
                    }
                )
                meta = {"goal": instruction, "steps": self._ui_elements_history}
                with open(
                    os.path.join(self.save_dir, "metadata.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Failed to save ui_elements metadata: {e}")
            try:
                screenshot_path = os.path.join(
                    self.save_dir, f"screenshot_step{self.turn_number - 1}.png"
                )
                Image.fromarray(screenshot).save(screenshot_path)
            except Exception as e:
                print(f"Failed to save screenshot: {e}")

    def _run_monitor(
        self,
        instruction: str,
        current_screenshot: np.ndarray,
    ) -> dict[str, Any]:
        recent_screens = list(self._recent_screenshots)[-self.monitor_num_recent_screens :]
        if not recent_screens:
            recent_screens = [current_screenshot]
        user_content = [
            {
                "type": "text",
                "text": (
                    f"Instruction: {instruction}\n"
                    f"Action history so far: {self.step_his or 'No action has been executed yet.'}\n"
                    "The following screenshots are ordered from older to newer. "
                    "The last screenshot is the current screen.\n"
                    "Decide whether the weak agent should continue acting on this step."
                ),
            }
        ]
        for img in recent_screens:
            user_content.append(
                {"type": "image_url", "image_url": {"url": self._to_base64_png(img)}}
            )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": MONITOR_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        raw_response = _chat_with_retry(
            self.monitor_client,
            self.model_name,
            messages,
            retry_times=3,
            retry_sleep_s=5.0,
            retry_name="monitor_model",
            request_timeout_s=60.0,
        )
        print("[monitor judge]")
        print(raw_response)
        print("=" * 50)
        decision = _parse_monitor_output(raw_response)
        decision["raw_response"] = raw_response
        decision["checked"] = True
        decision["screens_considered"] = len(recent_screens)
        return decision

    def _build_strong_messages(
        self,
        instruction: str,
        recent_screenshots: list[np.ndarray],
        intervention_feedback: str | None = None,
    ) -> list[dict[str, Any]]:
        if "gemini" in (self.model_name or "").lower():
            system_prompt = GEMINI3PRO_SYSTEM_PROMPT
            user_prompt = GEMINI3PRO_USER_PROMPT.format(
                instruction=instruction, history=self.step_his
            )
        else:
            system_prompt = QWEN3VL_SYSTEM_PROMPT
            user_prompt = QWEN3VL_USER_PROMPT.format(
                instruction=instruction, history=self.step_his
            )
        if intervention_feedback:
            user_prompt += (
                "\n\nIntervention context:\n"
                "The weak agent's recent execution was judged to be off track. "
                "You are now taking over from the current screen.\n"
                f"{intervention_feedback}\n"
                "Use this assessment as additional context, but ground your next action in the current screen and full history."
            )
        user_content = [{"type": "text", "text": user_prompt}]
        for img in recent_screenshots:
            user_content.append(
                {"type": "image_url", "image_url": {"url": self._to_base64_png(img)}}
            )
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _build_weak_messages(
        self,
        instruction: str,
        recent_screenshots: list[np.ndarray],
        width: int,
        height: int,
    ) -> tuple[list[dict[str, Any]], int, int]:
        resized_height, resized_width = smart_resize(
            height,
            width,
            min_pixels=self.weak_min_pixels,
            max_pixels=self.weak_max_pixels,
        )
        system_prompt = Qwen25VL_SYSTEM_PROMPT.format(
            resolution=f"{resized_width}x{resized_height}"
        )
        user_prompt = QWEN25VL_USER_PROMPT.format(
            instruction=instruction, history=self.step_his
        )
        user_content = [{"type": "text", "text": user_prompt}]
        for img in recent_screenshots:
            user_content.append(
                {"type": "image_url", "image_url": {"url": self._to_base64_png(img)}}
            )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        return messages, resized_width, resized_height

    def _run_strong_actor(
        self, instruction: str, recent_screenshots: list[np.ndarray], width: int, height: int
    ) -> tuple[str, dict[str, Any], str]:
        intervention_feedback = self.pending_intervention_feedback
        if intervention_feedback:
            print("[intervention feedback injected]")
            print(intervention_feedback)
            print("=" * 50)
        messages = self._build_strong_messages(
            instruction,
            recent_screenshots,
            intervention_feedback=intervention_feedback,
        )
        self.pending_intervention_feedback = None
        response = _chat_with_retry(
            self.strong_client,
            self.model_name,
            messages,
            retry_times=5,
            retry_sleep_s=10.0,
            retry_name="strong_model",
            request_timeout_s=60.0,
        )
        print("[strong response]")
        print(response)
        print("=" * 50)
        tool_call = _parse_tool_call_json(response)
        if not tool_call:
            raise ValueError("No <tool_call> JSON found in strong model output.")
        args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}
        parsed = qwen3vl_action_transform(args.get("action", ""), args, width, height)
        if "gemini" in (self.model_name or "").lower():
            conclusion_text = _extract_conclusion_text_qwen3vl_gemini(response)
            thinking_text = _extract_action_text_qwen3vl_gemini(response)
            op_text = conclusion_text if conclusion_text else thinking_text
        else:
            op_text = _extract_action_text_qwen3vl(response)
        return response, parsed, op_text

    def _run_weak_actor(
        self, instruction: str, recent_screenshots: list[np.ndarray], width: int, height: int
    ) -> tuple[str, dict[str, Any], str]:
        messages, resized_width, resized_height = self._build_weak_messages(
            instruction, recent_screenshots, width, height
        )
        response = _chat_with_retry(
            self.weak_client,
            self.weak_model_name,
            messages,
            retry_times=5,
            retry_sleep_s=10.0,
            retry_name="weak_model",
            request_timeout_s=60.0,
        )
        print("[weak response]")
        print(response)
        print("=" * 50)
        tool_call = _parse_tool_call_json(response)
        if not tool_call:
            raise ValueError("No <tool_call> JSON found in weak model output.")
        args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}
        parsed = qwen25vl_action_transform(
            args.get("action", ""),
            args,
            width,
            height,
            resized_width,
            resized_height,
        )
        tool_call_qwen3 = _convert_qwen25_tool_call_to_qwen3(
            tool_call,
            resized_width,
            resized_height,
        )
        response = _replace_tool_call_block(response, tool_call_qwen3)
        conclusion_text = _extract_conclusion_qwen25vl(response)
        thinking_text = _extract_thinking_qwen25vl(response)
        op_text = conclusion_text if conclusion_text else thinking_text
        return response, parsed, op_text

    @staticmethod
    def _action_signature_from_parsed(parsed: dict[str, Any]) -> str:
        try:
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(parsed)

    def step(self, instruction: str) -> base_agent.AgentInteractionResult:
        self.turn_number += 1
        print("[user instruction]")
        print(instruction)
        print("[current history]")
        print(self.step_his if self.step_his else "No action has been executed yet.")
        print("=" * 50)

        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()
        self._save_step_artifacts(instruction, state, screenshot)
        self._recent_screenshots.append(screenshot)

        monitor_output = {
            "checked": False,
            "should_intervene": False,
            "reason": "",
            "summary": "",
        }
        intervention_triggered = False
        if (
            self.current_policy == "weak"
            and self.should_monitor_next_step
            and self.intervention_count < self.max_interventions
        ):
            print("calling monitor...")
            monitor_output = self._run_monitor(instruction, screenshot)
            if bool(monitor_output.get("should_intervene")):
                summary = str(monitor_output.get("summary", "") or "").strip()
                reason = str(monitor_output.get("reason", "") or "").strip()
                feedback_parts = []
                if summary:
                    feedback_parts.append(f"Monitor summary: {summary}")
                if reason:
                    feedback_parts.append(f"Monitor reason: {reason}")
                self.pending_intervention_feedback = "\n".join(feedback_parts).strip()
                self.current_policy = "strong"
                self.has_intervened = True
                self.intervention_count += 1
                self.strong_steps_since_intervention = 0
                self.should_monitor_next_step = False
                intervention_triggered = True
                print("[switching] monitor triggered intervention, switching to strong actor")

        recent_screenshots = list(self._recent_screenshots)[-self.last_N :]
        height, width = screenshot.shape[:2]
        step_policy_source = self.current_policy

        try:
            if step_policy_source == "strong":
                response, parsed, op_text = self._run_strong_actor(
                    instruction, recent_screenshots, width, height
                )
            else:
                response, parsed, op_text = self._run_weak_actor(
                    instruction, recent_screenshots, width, height
                )
        except Exception as e:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": f"Failed to obtain action from {step_policy_source} actor: {e}",
                    "step_history": self.step_his,
                    "policy_source": step_policy_source,
                    "monitor_output": monitor_output,
                    "intervention_triggered": intervention_triggered,
                },
            )

        if op_text:
            self.step_his += f"Step {self.turn_number}: {op_text}; "

        if parsed.get("action_type") == "answer":
            try:
                act = json_action.JSONAction(**parsed)
                self.env.execute_action(act)
            except Exception:
                print("Failed to execute answer action:", parsed)
            return base_agent.AgentInteractionResult(
                True,
                {
                    "response": response,
                    "step_history": self.step_his,
                    "parsed": parsed,
                    "policy_source": step_policy_source,
                    "monitor_output": monitor_output,
                    "intervention_triggered": intervention_triggered,
                },
            )

        action_sig = self._action_signature_from_parsed(parsed)
        if self.last_action == action_sig:
            self.repeat_time += 1
        else:
            self.repeat_time = 0
        self.last_action = action_sig

        action_executed = False
        try:
            act = json_action.JSONAction(**parsed)
            self.env.execute_action(act)
            time.sleep(self.wait_after_action_seconds)
            action_executed = True
        except Exception:
            print("Failed to execute action:", parsed)

        if action_executed and step_policy_source == "strong":
            self.strong_steps_since_intervention += 1
            self.should_monitor_next_step = False
            if (
                self.max_interventions >= 2
                and self.intervention_count == 1
                and self.strong_steps_since_intervention
                >= self.strong_min_steps_after_intervention
            ):
                self.current_policy = "weak"
                self.strong_steps_since_intervention = 0
                print(
                    "[switching] first strong intervention finished, switching back to weak actor"
                )
        elif action_executed and step_policy_source == "weak":
            self.should_monitor_next_step = True

        if parsed.get("action_type") == "status":
            return base_agent.AgentInteractionResult(
                True,
                {
                    "response": response,
                    "step_history": self.step_his,
                    "parsed": parsed,
                    "policy_source": step_policy_source,
                    "monitor_output": monitor_output,
                    "intervention_triggered": intervention_triggered,
                },
            )

        if self.repeat_time >= 10:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": "Terminated due to repeated identical actions.",
                    "response": response,
                    "step_history": self.step_his,
                    "parsed": parsed,
                    "repeat_time": self.repeat_time,
                    "policy_source": step_policy_source,
                    "monitor_output": monitor_output,
                    "intervention_triggered": intervention_triggered,
                },
            )

        return base_agent.AgentInteractionResult(
            False,
            {
                "response": response,
                "step_history": self.step_his,
                "parsed": parsed,
                "policy_source": step_policy_source,
                "monitor_output": monitor_output,
                "intervention_triggered": intervention_triggered,
            },
        )



def _extract_thinking_qwen25vl(response: str) -> str:
    """Extract content from <thinking>...</thinking> tags."""
    m = re.search(r"<thinking>\s*([\s\S]*?)\s*</thinking>", response)
    return m.group(1).strip() if m else ""


def _extract_conclusion_qwen25vl(response: str) -> str:
    """Extract content from <conclusion>...</conclusion> tags."""
    m = re.search(r"<conclusion>\s*([\s\S]*?)\s*</conclusion>", response)
    return m.group(1).strip() if m else ""


class Qwen25VL(base_agent.EnvironmentInteractingAgent):
    """Android GUI Agent based on Qwen2.5VL tool-call output (for AndroidWorld eval).

    - Input: Screenshot + instruction + history
    - Output: <thinking>...</thinking><tool_call>{...}</tool_call><conclusion>...</conclusion>
    - Execution: Map to JSONAction by qwen25vl_action_transform(...)
    
    Key differences from Qwen3VL:
    - Uses actual pixel coordinates (via smart_resize) instead of normalized 0-1000
    - Includes thinking/conclusion tags for step history
    - Resolution is passed in system prompt
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "Qwen25VL",
        wait_after_action_seconds: float = 2.0,
        model_base_url: str = "http://127.0.0.1:8000/v1",
        model_api_key: str = "EMPTY",
        model_name: str = "",
        extra_headers: dict[str, str] | None = None,
        min_pixels: int = 3136,
        max_pixels: int = 12845056,
    ):
        super().__init__(env, name)
        self.llm = llm
        self.wait_after_action_seconds = wait_after_action_seconds
        self.model_name = model_name
        self.client = OpenAI(
            api_key=model_api_key,
            base_url=model_base_url,
            default_headers=extra_headers,
        )

        # Used for self-deployed model (Not Used)
        self.model_base_url = model_base_url
        
        self.step_his: str = ""
        self.turn_number: int = 0

        # Provide multiple most recent screenshots to the model (hard-coded; user may adjust).
        self.last_N = 1
        self._recent_screenshots = deque(maxlen=self.last_N)

        # Used to detect repeated actions (avoid infinite loops)
        self.last_action: str | None = None
        self.repeat_time: int = 0

        # Qwen2.5VL specific: smart_resize parameters
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Per-step ui_elements metadata aligned with screenshot_step{step}.png.
        self._ui_elements_history: list[dict[str, Any]] = []

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self.step_his = ""
        self.turn_number = 0
        self._recent_screenshots.clear()
        self.last_action = None
        self.repeat_time = 0
        self._ui_elements_history = []

    @staticmethod
    def _to_base64_png(image: np.ndarray) -> str:
        import base64
        from io import BytesIO
        from PIL import Image as PILImage
        buf = BytesIO()
        PILImage.fromarray(image).save(buf, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def step(self, instruction: str) -> base_agent.AgentInteractionResult:
        self.turn_number += 1

        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()  # RGB format from Android

        # Save per-step ui_elements metadata for post-processing (e.g., bbox recovery).
        if self.save_dir is not None:
            try:
                step_idx = self.turn_number - 1
                self._ui_elements_history.append(
                    {
                        "step": step_idx,
                        "logical_screen_size": list(self.env.logical_screen_size),
                        "ui_elements": [
                            _ui_element_to_metadata_dict(e) for e in state.ui_elements
                        ],
                    }
                )
                meta = {"goal": instruction, "steps": self._ui_elements_history}
                with open(
                    os.path.join(self.save_dir, "metadata.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Failed to save ui_elements metadata: {e}")

        # Save screenshot
        if self.save_dir is not None:
            try:
                screenshot_path = os.path.join(
                    self.save_dir, f"screenshot_step{self.turn_number - 1}.png"
                )
                Image.fromarray(screenshot).save(screenshot_path)
            except Exception as e:
                print(f"Failed to save screenshot: {e}")

        self._recent_screenshots.append(screenshot)
        height, width = screenshot.shape[:2]

        # Compute resized dimensions using smart_resize
        resized_height, resized_width = smart_resize(
            height, width,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        # Format resolution string for prompt
        resolution_str = f"{resized_width}x{resized_height}"

        # Use Qwen25VL specific prompts
        system_prompt = Qwen25VL_SYSTEM_PROMPT.format(resolution=resolution_str)
        user_prompt = QWEN25VL_USER_PROMPT.format(
            instruction=instruction, history=self.step_his
        )
        print(user_prompt)

        # Build message with multiple recent screenshots
        user_content = [{"type": "text", "text": user_prompt}]
        for img in list(self._recent_screenshots):
            user_content.append(
                {"type": "image_url", "image_url": {"url": self._to_base64_png(img)}}
            )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        response = ""
        completion = None
        for _ in range(5):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    timeout=60.0,
                )
            except Exception as e:
                print(f"Qwen25VL request failed or timed out after 60s: {e}")
                response = ""
                continue
            # print(completion)
            try:
                response = completion.choices[0].message.content or ""
            except Exception:
                response = ""
            if response.strip():
                break
            print("sleep 10 seconds and retry")
            time.sleep(10)

        print(response)
        print("=" * 50)

        tool_call = _parse_tool_call_json(response)
        if not tool_call:
            return base_agent.AgentInteractionResult(
                True, {"summary": "No <tool_call> JSON found in model output.", "response": response}
            )

        # Extract thinking and conclusion for history
        thinking_text = _extract_thinking_qwen25vl(response)
        conclusion_text = _extract_conclusion_qwen25vl(response)
        
        # Build step history: prefer conclusion, fallback to thinking + tool_call
        if conclusion_text:
            step_summary = conclusion_text
        elif thinking_text:
            step_summary = thinking_text
        else:
            step_summary = ""

        if step_summary:
            self.step_his += f"Step {self.turn_number}: {step_summary}; "

        # Compatible: tool_call may look like {"name":"mobile_use","arguments":{...}}
        args = tool_call.get("arguments", {}) if isinstance(tool_call, dict) else {}
        action_name = args.get("action", "")
        try:
            parsed = qwen25vl_action_transform(
                action_name, args, width, height, resized_width, resized_height
            )
            print(parsed)
        except Exception as e:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": f"Failed to transform tool-call into action: {e}",
                    "response": response,
                    "tool_call": tool_call,
                },
            )

        # If model outputs an answer, persist it and stop immediately.
        if parsed.get("action_type") == "answer":
            try:
                act = json_action.JSONAction(**parsed)
                self.env.execute_action(act)
            except Exception:
                print("Failed to execute answer action:", parsed)
            return base_agent.AgentInteractionResult(
                True, {"response": response, "step_history": self.step_his, "parsed": parsed}
            )

        # Record last_action + repeat_time
        try:
            action_sig = json.dumps(args, ensure_ascii=False, sort_keys=True)
        except Exception:
            action_sig = str(args)
        if self.last_action == action_sig:
            self.repeat_time += 1
        else:
            self.repeat_time = 0
        self.last_action = action_sig

        try:
            act = json_action.JSONAction(**parsed)
            self.env.execute_action(act)
            time.sleep(self.wait_after_action_seconds)
        except Exception:
            print("Failed to execute action:", parsed)

        if parsed.get("action_type") == "status":
            return base_agent.AgentInteractionResult(
                True, {"response": response, "step_history": self.step_his, "parsed": parsed}
            )

        # If repeated actions reach the threshold: terminate immediately to avoid deadlock
        if self.repeat_time >= 10:
            return base_agent.AgentInteractionResult(
                True,
                {
                    "summary": "Terminated due to repeated identical actions.",
                    "response": response,
                    "step_history": self.step_his,
                    "parsed": parsed,
                    "repeat_time": self.repeat_time,
                },
            )

        return base_agent.AgentInteractionResult(
            False, {"response": response, "step_history": self.step_his, "parsed": parsed}
        )

