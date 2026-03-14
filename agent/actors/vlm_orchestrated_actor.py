"""VLM orchestrated actor with multi-step planning."""

import json
from collections.abc import Callable
from typing import cast, Callable
import uuid
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import copy
from pathlib import Path
from datetime import datetime
from anthropic import APIResponse
from anthropic.types import ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage

from actors.llm.openai_client import run_oai_interleaved
from actors.llm.groq_client import run_groq_interleaved
from actors.llm.utils import is_image_path
import time
import re
import os

OUTPUT_DIR = "./tmp/outputs"

ORCHESTRATOR_LEDGER_PROMPT = """
Recall we are working on the following request:

{task}

To make progress on the request, please answer the following questions, including necessary reasoning:

    - Is the request fully satisfied? (True if complete, or False if the original request has yet to be SUCCESSFULLY and FULLY addressed)
    - Are we in a loop where we are repeating the same requests and / or getting the same responses as before?
    - Are we making forward progress?
    - What instruction or question would you give in order to complete the task?

Please output an answer in pure JSON format according to the following schema:

    {{
       "is_request_satisfied": {{
            "reason": string,
            "answer": boolean
        }},
        "is_in_loop": {{
            "reason": string,
            "answer": boolean
        }},
        "is_progress_being_made": {{
            "reason": string,
            "answer": boolean
        }},
        "instruction_or_question": {{
            "reason": string,
            "answer": string
        }}
    }}
"""


def extract_data(input_string, data_type):
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    matches = re.findall(pattern, input_string, re.DOTALL)
    return matches[0][0].strip() if matches else input_string


class VLMOrchestratedAgent:
    def __init__(
        self,
        model: str,
        provider: str,
        api_key: str,
        output_callback: Callable,
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
        save_folder: str = None,
    ):
        if model == "omniparser + gpt-4o" or model == "omniparser + gpt-4o-orchestrated":
            self.model = "gpt-4o-2024-11-20"
        elif model == "omniparser + R1" or model == "omniparser + R1-orchestrated":
            self.model = "deepseek-r1-distill-llama-70b"
        elif model == "omniparser + qwen2.5vl" or model == "omniparser + qwen2.5vl-orchestrated":
            self.model = "qwen2.5-vl-72b-instruct"
        elif model == "omniparser + o1" or model == "omniparser + o1-orchestrated":
            self.model = "o1"
        elif model == "omniparser + o3-mini" or model == "omniparser + o3-mini-orchestrated":
            self.model = "o3-mini"
        else:
            raise ValueError(f"Model {model} not supported")

        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback
        self.save_folder = save_folder
        if self.save_folder:
            os.makedirs(self.save_folder, exist_ok=True)
        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0
        self.step_count = 0
        self.plan, self.ledger = None, None
        self.system = ''

    def __call__(self, messages: list, parsed_screen: list[str, list, dict]):
        if self.step_count == 0:
            plan = self._initialize_task(messages)
            self.output_callback(f'-- Plan: {plan} --')
            messages.append({"role": "assistant", "content": plan})
        else:
            updated_ledger = self._update_ledger(messages)
            self.output_callback(
                f'<details><summary><strong>Task Progress Ledger</strong></summary>'
                f'<pre>{updated_ledger}</pre></details>'
            )
            messages.append({"role": "assistant", "content": updated_ledger})
            self.ledger = updated_ledger

        self.step_count += 1
        with open(f"{self.save_folder}/screenshot_{self.step_count}.png", "wb") as f:
            f.write(base64.b64decode(parsed_screen['original_screenshot_base64']))
        with open(f"{self.save_folder}/som_screenshot_{self.step_count}.png", "wb") as f:
            f.write(base64.b64decode(parsed_screen['som_image_base64']))

        latency_omniparser = parsed_screen['latency']
        screen_info = str(parsed_screen['screen_info'])
        screenshot_uuid = parsed_screen['screenshot_uuid']
        screen_width, screen_height = parsed_screen['width'], parsed_screen['height']

        boxids_and_labels = parsed_screen["screen_info"]
        system = self._get_system_prompt(boxids_and_labels)

        planner_messages = messages
        _remove_som_images(planner_messages)
        _maybe_filter_to_n_most_recent_images(planner_messages, self.only_n_most_recent_images)

        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_{screenshot_uuid}.png")
            planner_messages[-1]["content"].append(f"{OUTPUT_DIR}/screenshot_som_{screenshot_uuid}.png")

        start = time.time()
        if "gpt" in self.model or "o1" in self.model or "o3-mini" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages, system=system, model_name=self.model,
                api_key=self.api_key, max_tokens=self.max_tokens,
                provider_base_url="https://api.openai.com/v1", temperature=0,
            )
            self.total_token_usage += token_usage
            if 'gpt' in self.model:
                self.total_cost += (token_usage * 2.5 / 1000000)
            elif 'o1' in self.model:
                self.total_cost += (token_usage * 15 / 1000000)
            elif 'o3-mini' in self.model:
                self.total_cost += (token_usage * 1.1 / 1000000)
        elif "r1" in self.model:
            vlm_response, token_usage = run_groq_interleaved(
                messages=planner_messages, system=system, model_name=self.model,
                api_key=self.api_key, max_tokens=self.max_tokens,
            )
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 0.99 / 1000000)
        elif "qwen" in self.model:
            vlm_response, token_usage = run_oai_interleaved(
                messages=planner_messages, system=system, model_name=self.model,
                api_key=self.api_key, max_tokens=min(2048, self.max_tokens),
                provider_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0,
            )
            self.total_token_usage += token_usage
            self.total_cost += (token_usage * 2.2 / 1000000)
        else:
            raise ValueError(f"Model {self.model} not supported")
        latency_vlm = time.time() - start
        self.output_callback(f'<i>Step {self.step_count} | Parser: {latency_omniparser:.2f}s | LLM: {latency_vlm:.2f}s</i>')

        if self.print_usage:
            print(f"Total token: {self.total_token_usage}. Cost: ${self.total_cost:.5f}")

        vlm_response_json = extract_data(vlm_response, "json")
        vlm_response_json = json.loads(vlm_response_json)

        img_to_show_base64 = parsed_screen["som_image_base64"]
        if "Box ID" in vlm_response_json:
            try:
                bbox = parsed_screen["parsed_content_list"][int(vlm_response_json["Box ID"])]["bbox"]
                vlm_response_json["box_centroid_coordinate"] = [int((bbox[0] + bbox[2]) / 2 * screen_width), int((bbox[1] + bbox[3]) / 2 * screen_height)]
                img_to_show_data = base64.b64decode(img_to_show_base64)
                img_to_show = Image.open(BytesIO(img_to_show_data))
                draw = ImageDraw.Draw(img_to_show)
                x, y = vlm_response_json["box_centroid_coordinate"]
                radius = 10
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
                draw.ellipse((x - radius*3, y - radius*3, x + radius*3, y + radius*3), fill=None, outline='red', width=2)
                buffered = BytesIO()
                img_to_show.save(buffered, format="PNG")
                img_to_show_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            except:
                pass
        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">')
        self.output_callback(
            f'<details><summary><strong>Parsed Screen Elements</strong></summary>'
            f'<pre>{screen_info}</pre></details>'
        )

        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'

        response_content = [BetaTextBlock(text=vlm_plan_str, type='text')]
        if 'box_centroid_coordinate' in vlm_response_json:
            move_cursor_block = BetaToolUseBlock(
                id=f'toolu_{uuid.uuid4()}',
                input={'action': 'mouse_move', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                name='computer', type='tool_use',
            )
            response_content.append(move_cursor_block)

        if vlm_response_json["Next Action"] == "None":
            print("Task paused/completed.")
        elif vlm_response_json["Next Action"] == "type":
            sim_content_block = BetaToolUseBlock(
                id=f'toolu_{uuid.uuid4()}',
                input={'action': vlm_response_json["Next Action"], 'text': vlm_response_json["value"]},
                name='computer', type='tool_use',
            )
            response_content.append(sim_content_block)
        else:
            sim_content_block = BetaToolUseBlock(
                id=f'toolu_{uuid.uuid4()}',
                input={'action': vlm_response_json["Next Action"]},
                name='computer', type='tool_use',
            )
            response_content.append(sim_content_block)
        response_message = BetaMessage(
            id=f'toolu_{uuid.uuid4()}', content=response_content,
            model='', role='assistant', type='message',
            stop_reason='tool_use', usage=BetaUsage(input_tokens=0, output_tokens=0),
        )

        step_trajectory = {
            "screenshot_path": f"{self.save_folder}/screenshot_{self.step_count}.png",
            "som_screenshot_path": f"{self.save_folder}/som_screenshot_{self.step_count}.png",
            "screen_info": screen_info,
            "latency_omniparser": latency_omniparser,
            "latency_vlm": latency_vlm,
            "vlm_response_json": vlm_response_json,
            'ledger': self.ledger,
        }
        with open(f"{self.save_folder}/trajectory.json", "a") as f:
            f.write(json.dumps(step_trajectory))
            f.write("\n")

        return response_message, vlm_response_json

    def _get_system_prompt(self, screen_info: str = ""):
        main_section = f"""
You are using a computer.
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
You can only interact with the desktop GUI (no terminal or application menu access).

You may be given some history plan and actions, this is the response from the previous loop.
You should carefully consider your plan base on the task, screenshot, and history actions.

Here is the list of all detected bounding boxes by IDs on the screen and their description:{screen_info}

Your available "Next Action" only include:
- type: types a string of text.
- left_click: move mouse to box id and left clicks.
- right_click: move mouse to box id and right clicks.
- double_click: move mouse to box id and double clicks.
- hover: move mouse to box id.
- scroll_up: scrolls the screen up to view previous content.
- scroll_down: scrolls the screen down.
- wait: waits for 1 second for the device to load or respond.

Output format:
```json
{{
    "Reasoning": str,
    "Next Action": "action_type, action description" | "None"
    "Box ID": n,
    "value": "xxx"
}}
```

IMPORTANT NOTES:
1. You should only give a single action at a time.
"""
        thinking_model = "r1" in self.model
        if not thinking_model:
            main_section += "\n2. Analyze the current screen and reflect on history.\n"
        else:
            main_section += "\n2. In <think> tags analyze the screen. In <output> tags put the JSON.\n"
        main_section += """
3. Attach the next action prediction in the "Next Action".
4. No keyboard shortcuts.
5. When done, say "Next Action": "None".
6. Break tasks into subgoals.
7. Avoid repeating the same action.
8. If prompted with login/captcha, say "Next Action": "None".
"""
        return main_section

    def _initialize_task(self, messages: list):
        self._task = messages[0]["content"]
        plan_prompt = f"""
        please devise a short bullet-point plan for addressing the original user task: {self._task}
        You should write your plan in a json dict, e.g:```json
{{
'step 1': xxx,
'step 2': xxxx,
...
}}```
        Now start your answer directly.
        """
        input_message = copy.deepcopy(messages)
        input_message.append({"role": "user", "content": plan_prompt})
        vlm_response, token_usage = run_oai_interleaved(
            messages=input_message, system="", model_name=self.model,
            api_key=self.api_key, max_tokens=self.max_tokens,
            provider_base_url="https://api.openai.com/v1", temperature=0,
        )
        plan = extract_data(vlm_response, "json")
        plan_path = os.path.join(self.save_folder, "plan.json")
        try:
            with open(plan_path, "w") as f:
                f.write(plan)
        except Exception as e:
            print(f"Error saving plan: {e}")
        return plan

    def _update_ledger(self, messages):
        update_ledger_prompt = ORCHESTRATOR_LEDGER_PROMPT.format(task=self._task)
        input_message = copy.deepcopy(messages)
        input_message.append({"role": "user", "content": update_ledger_prompt})
        vlm_response, token_usage = run_oai_interleaved(
            messages=input_message, system="", model_name=self.model,
            api_key=self.api_key, max_tokens=self.max_tokens,
            provider_base_url="https://api.openai.com/v1", temperature=0,
        )
        return extract_data(vlm_response, "json")


def _remove_som_images(messages):
    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg["content"] = [
                cnt for cnt in msg_content
                if not (isinstance(cnt, str) and 'som' in cnt and is_image_path(cnt))
            ]


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    if images_to_keep is None:
        return messages

    total_images = 0
    for msg in messages:
        for cnt in msg.get("content", []):
            if isinstance(cnt, str) and is_image_path(cnt):
                total_images += 1
            elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                for content in cnt.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        total_images += 1

    images_to_remove = total_images - images_to_keep

    for msg in messages:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            new_content = []
            for cnt in msg_content:
                if isinstance(cnt, str) and is_image_path(cnt):
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                elif isinstance(cnt, dict) and cnt.get("type") == "tool_result":
                    new_tool_result_content = []
                    for tool_result_entry in cnt.get("content", []):
                        if isinstance(tool_result_entry, dict) and tool_result_entry.get("type") == "image":
                            if images_to_remove > 0:
                                images_to_remove -= 1
                                continue
                        new_tool_result_content.append(tool_result_entry)
                    cnt["content"] = new_tool_result_content
                new_content.append(cnt)
            msg["content"] = new_content
