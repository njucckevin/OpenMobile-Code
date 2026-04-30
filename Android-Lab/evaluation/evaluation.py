import templates.seeact_screenshot_prompts as SeeActPrompts
from evaluation.definition import *
from evaluation.utils import *
from templates import *
from agent.utils import *
from page_executor.utils import smart_resize
import time
import json
class AutoTask():
    def __init__(self, instruction, controller, page_executor, agent, record, command_per_step, **kwargs):
        self.controller = controller
        self.page_executor = page_executor
        self.agent = agent
        self.record = record
        self.kwargs = kwargs
        self.set_system_prompt(instruction)
        self.actions = ""
        self.instruction = instruction
        self.record.command_per_step = [command_per_step]
        if "map.me" in instruction or "pimusic" in instruction:
            self.accessibility = self.controller.check_ac_survive()
        else:
            self.accessibility = False

    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": self.agent.system_prompt(instruction)
        }]

    def run_step(self, round_count):
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility)
        compressed_xml_json = self.record.get_latest_xml()

        prompt = f"" if round_count == 0 else "** XML **\n"
        try:
            current_message = {"role": "user", "content": prompt + compressed_xml_json}
            if self.agent.name == "GLMModelAgent":
                current_message["current_app"] = self.controller.get_current_activity()
            rsp = self.agent.act([*self.record.history, current_message])
        except Exception as e:
            print_with_color(f"Error: {e}", "red")

        exe_res = self.page_executor(get_code_snippet(rsp))
        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1


class TextOnlyTask(AutoTask):
    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": SYSTEM_PROMPT_ANDROID_TEXT_GPT + f"\n\nTask Instruction: {instruction}"
        }]


class ScreenshotTask(TextOnlyTask):
    def run_step(self, round_count):
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility,
                                  need_labeled=True)
        prompt = f"" if round_count == 0 else "** XML **\n"
        try:
            xml = self.record.get_latest_xml()
            image_path = self.record.labeled_current_screenshot_path
            current_message = self.agent.prompt_to_message(prompt, [image_path])
            rsp = self.agent.act([*self.record.history, current_message])
        except Exception as e:
            import traceback
            print_with_color(f"Error: {e}", "red")

        exe_res = self.page_executor(get_code_snippet(rsp))
        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1

    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": SYSTEM_PROMPT_ANDROID_MLLM_DIRECT + f"\n\nTask Instruction: {instruction}"
        }]

class PixellevelTask(TextOnlyTask): 
    def run_step(self, round_count):
        self.system_prompt = {
            "role": "system",
            "content": SYSTEM_PROMPT_ANDROID_MLLM_PixelLevel
        }
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility,
                                  need_labeled=False)
        prompt = f'''{self.instruction}'''
        print(f"Prompt: {prompt}")
        try:
            image_path = self.page_executor.current_screenshot
            current_message = self.agent.prompt_to_message(prompt, [image_path])
            rsp = self.agent.act([self.system_prompt, *self.record.history, current_message])
        except Exception as e:
            import traceback
            print_with_color(f"Error: {e}", "red")
        print(rsp)
        exe_res = self.page_executor(get_code_snippet(rsp))
        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1
    def set_system_prompt(self, instruction):
        print(f"Setting system prompt: {SYSTEM_PROMPT_ANDROID_MLLM_PixelLevel}")
        self.record.history = [{
            "role": "system",
            "content": SYSTEM_PROMPT_ANDROID_MLLM_PixelLevel
        }]

class CogAgentTask(TextOnlyTask):
    def run_step(self, round_count):
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility,
                                  need_labeled=True)
        prompt = f"" if round_count == 0 else json.dumps({"current_app": self.controller.get_current_app()},
                                                         ensure_ascii=False)
        try:
            image_path = self.page_executor.current_screenshot
            current_message = self.agent.prompt_to_message(prompt, [image_path])
            rsp = self.agent.act([*self.record.history, current_message])
        except Exception as e:
            import traceback
            print(traceback.print_exc())

        exe_res = self.page_executor(get_code_snippet(rsp))
        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1

    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": SYSTEM_PROMPT_ANDROID_MLLM_CogAgent + f"\n\nTask Instruction: {instruction}"
        }]


class ScreenshotReactTask(ScreenshotTask):
    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": SYSTEM_PROMPT_ANDROID_MLLM_DIRECT_REACT + f"\n\nTask Instruction: {instruction}"
        }]


class ScreenSeeActTask(TextOnlyTask):

    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": SeeActPrompts.QUERY_SYSTEM_PROMPT
        }]
        self.stage_one_record = []
        self.instruction = instruction

    def run_step(self, round_count):
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility,
                                  need_labeled=False)
        try:
            xml_tree = self.record.get_latest_xml_tree()
            choices_list = extract_bounds(xml_tree)
            image_path = self.page_executor.current_screenshot
            system_prompt = SeeActPrompts.QUERY_SYSTEM_PROMPT
            query_user_prompt = SeeActPrompts.QUERY_USER_PROMPT.format(
                task=self.instruction,
                previous_actions=("\n\n".join(self.stage_one_record) or "None")
            )
            query_message = self.agent.prompt_to_message(query_user_prompt, [image_path])
            referring_user_prompt = SeeActPrompts.REFERRING_USER_PROMPT.format(
                option_prompt="\n".join(f"{item['key']} | {item['value']}" for item in choices_list)
            )

            messages = [
                {"role": "system", "content": system_prompt},
                query_message,
            ]

            print(">> Stage 1. Query")
            with open("monitor.log", "w") as f:
                f.write(json.dumps(messages, indent=4))
            description = self.agent.act(messages)
            print(description, end="\n\n")
            with open("monitor.log", "w") as f:
                f.write(description)
            messages.append({"role": "assistant", "content": description})
            messages.append({"role": "user", "content": referring_user_prompt})

            print(">> Stage 2. Referring")
            with open("monitor.log", "w") as f:
                f.write(json.dumps(messages, indent=4))

            referring = self.agent.act(messages)
            print(referring, end="\n\n")
            with open("monitor.log", "w") as f:
                f.write(referring)


        except Exception as e:
            import traceback
            print(traceback.print_exc())
        referring = referring.split("Final Answer:")[-1].strip()
        exe_res = self.page_executor(get_code_snippet(referring))
        self.stage_one_record.append(description)
        self.record.update_after(exe_res, description + "\n\n==========\n\n" + referring)
        self.record.turn_number += 1


class TextOnlyReactTask(TextOnlyTask):
    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": SYSTEM_PROMPT_ANDROID_TEXT_ReAct + f"\n\nTask Instruction: {instruction}"
        }]


class TextOnlyFineTuneTask(TextOnlyTask):
    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": SYSTEM_PROMPT_ANDROID_TEXT_GLM_v1_5 + f"\n\nTask Instruction: {instruction}"
        }]

    def run_step(self, round_count):
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility)
        compressed_xml_json = self.record.get_latest_xml()

        try:
            app_info = f"{json.dumps({'current_app': self.controller.get_current_app()}, ensure_ascii=False)}\n"
            current_message = {"role": "user", "content": app_info + compressed_xml_json}
            rsp = self.agent.act([*self.record.history, current_message])
        except Exception as e:
            print_with_color(f"Error: {e}", "red")

        exe_res = self.page_executor(get_code_snippet(rsp))
        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1


class TextOnlyFineTuneTask_long(TextOnlyFineTuneTask):
    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": SYSTEM_PROMPT_ANDROID_TEXT_GPT + f"\n\nTask Instruction: {instruction}"
        }]

class PixellevelQwen3VLTask(TextOnlyTask): 
    def run_step(self, round_count):
        self.system_prompt = {
            "role": "system",
            "content": QWEN3VL_SYSTEM_PROMPT
        }
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility, need_labeled=False)
        prompt = QWEN3VL_USER_PROMPT.format(instruction=self.instruction, history=self.actions)
        try:
            image_path = self.page_executor.current_screenshot
            current_message = self.agent.prompt_to_message(prompt, [image_path])
            rsp, act_m, op_text = self.agent.act([self.system_prompt, current_message])
        except Exception as e:
            import traceback
            print_with_color(f"Error: {e}", "red")
        print(rsp)
        operation = op_text
        action = act_m
        self.actions += f"Step {self.record.turn_number}: {operation}; "
        exe_res = self.page_executor(get_code_snippet(action))
        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1
        
    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": QWEN3VL_SYSTEM_PROMPT
        }]

class PixellevelQwen2d5VLTask(TextOnlyTask): 
    def run_step(self, round_count):
        self.system_prompt = {
            "role": "system",
            "content": QWEN2D5VL_SYSTEM_PROMPT
        }
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility, need_labeled=False)
        prompt = QWEN2D5VL_USER_PROMPT.format(Instruction=self.instruction, agent_his=self.actions)
        try:
            image_path = self.page_executor.current_screenshot
            current_message = self.agent.prompt_to_message(prompt, [image_path])
            rsp, act_m, op_text = self.agent.act([self.system_prompt, current_message])
        except Exception as e:
            import traceback
            print_with_color(f"Error: {e}", "red")
        print(rsp)
        operation = op_text
        action = act_m
        self.actions += f"Step {self.record.turn_number}: {operation}; "
        print(f"--------action:{operation}")
        self.page_executor.image_height, self.page_executor.image_width = smart_resize(height=self.page_executor.viewport_height, width=self.page_executor.viewport_width, min_pixels=3136, max_pixels=21097440)
        print(self.page_executor.image_height, self.page_executor.image_width, self.page_executor.viewport_height, self.page_executor.viewport_width)
        exe_res = self.page_executor(get_code_snippet(action))
        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1
        
    def set_system_prompt(self, instruction):
        self.record.history = [{
            "role": "system",
            "content": QWEN2D5VL_SYSTEM_PROMPT
        }]


class PixellevelScaleCUATask(TextOnlyTask): 
    def run_step(self, round_count):
        self.system_prompt = {
            "role": "system",
            "content": ScaleCUA_SYSTEM_PROMPT
        }

        self.record.update_before(
            controller=self.controller,
            need_screenshot=True,
            ac_status=self.accessibility,
            need_labeled=False
        )

        prompt = f'''Please generate the next move according to the UI screenshot, the task and previous operations.

Task:
{self.instruction}

Previous operations:
{self.actions}'''

        try:
            image_path = self.page_executor.current_screenshot
            current_message = self.agent.prompt_to_message(prompt, [image_path])
            
            rsp, act_m, op_text = self.agent.act([self.system_prompt, current_message])

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print_with_color(f"Error: {e}", "red")
            return

        print(rsp)

        sections = parse_sections(rsp)

        think = sections.get("think")
        operation = sections.get("operation") or op_text
        action = sections.get("action")

        self.actions += f"Step {self.record.turn_number}: {operation}\n"

        self.page_executor.image_height, self.page_executor.image_width = smart_resize(
            self.page_executor.viewport_height * 1.0,
            self.page_executor.viewport_width * 1.0,
            min_pixels=3136,
            max_pixels=2109744
        )

        exe_res = self.page_executor(get_code_snippet(act_m))

        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1
