"""VLM actor — sends parsed screen to any model via OpenRouter."""

import json
from collections.abc import Callable
from typing import cast, Callable
import uuid
from PIL import Image, ImageDraw
import base64
from io import BytesIO

from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam, BetaUsage

from actors.llm.openrouter_client import run_openrouter_interleaved
from actors.llm.utils import is_image_path
import time
import re

OUTPUT_DIR = "./tmp/outputs"


def extract_data(input_string, data_type):
    pattern = f"```{data_type}" + r"(.*?)(```|$)"
    matches = re.findall(pattern, input_string, re.DOTALL)
    return matches[0][0].strip() if matches else input_string


class VLMAgent:
    def __init__(
        self,
        model: str,
        api_key: str,
        output_callback: Callable,
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
    ):
        self.model = model
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.output_callback = output_callback
        self.print_usage = print_usage
        self.total_token_usage = 0
        self.step_count = 0
        self.system = ''

    def __call__(self, messages: list, parsed_screen: list[str, list, dict]):
        self.step_count += 1
        image_base64 = parsed_screen['original_screenshot_base64']
        latency_omniparser = parsed_screen['latency']
        self.output_callback(f'-- Step {self.step_count}: --', sender="bot")
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
        vlm_response, token_usage = run_openrouter_interleaved(
            messages=planner_messages, system=system, model_name=self.model,
            api_key=self.api_key, max_tokens=self.max_tokens, temperature=0,
        )
        self.total_token_usage += token_usage
        latency_vlm = time.time() - start
        self.output_callback(f"LLM: {latency_vlm:.2f}s, Parser: {latency_omniparser:.2f}s", sender="bot")

        if self.print_usage:
            print(f"Total tokens: {self.total_token_usage}")

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
        self.output_callback(f'<img src="data:image/png;base64,{img_to_show_base64}">', sender="bot")
        self.output_callback(
            f'<details><summary>Parsed Screen elements</summary><pre>{screen_info}</pre></details>',
            sender="bot",
        )
        vlm_plan_str = ""
        for key, value in vlm_response_json.items():
            if key == "Reasoning":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'

        response_content = [BetaTextBlock(text=vlm_plan_str, type='text')]

        # Extract just the action type (model may return "left_click, description")
        raw_action = vlm_response_json["Next Action"]
        action_type = raw_action.split(",")[0].strip()

        if action_type == "None":
            print("Task paused/completed.")
        else:
            # Only move cursor and execute actions when task is not done
            if 'box_centroid_coordinate' in vlm_response_json:
                move_cursor_block = BetaToolUseBlock(
                    id=f'toolu_{uuid.uuid4()}',
                    input={'action': 'mouse_move', 'coordinate': vlm_response_json["box_centroid_coordinate"]},
                    name='computer', type='tool_use',
                )
                response_content.append(move_cursor_block)

            if action_type == "type":
                sim_content_block = BetaToolUseBlock(
                    id=f'toolu_{uuid.uuid4()}',
                    input={'action': action_type, 'text': vlm_response_json["value"]},
                    name='computer', type='tool_use',
                )
                response_content.append(sim_content_block)
            else:
                sim_content_block = BetaToolUseBlock(
                    id=f'toolu_{uuid.uuid4()}',
                    input={'action': action_type},
                    name='computer', type='tool_use',
                )
                response_content.append(sim_content_block)
        response_message = BetaMessage(
            id=f'toolu_{uuid.uuid4()}', content=response_content,
            model='', role='assistant', type='message',
            stop_reason='tool_use', usage=BetaUsage(input_tokens=0, output_tokens=0),
        )
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
- scroll_down: scrolls the screen down, when the desired button is not visible, or you need to see more content.
- wait: waits for 1 second for the device to load or respond.

Based on the visual information from the screenshot image and the detected bounding boxes, please determine the next action, the Box ID you should operate on (if action is one of 'type', 'hover', 'scroll_up', 'scroll_down', 'wait', there should be no Box ID field), and the value (if the action is 'type') in order to complete the task.

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
2. You should give an analysis to the current screen, and reflect on what has been done by looking at the history.
3. Attach the next action prediction in the "Next Action".
4. You should not include other actions, such as keyboard shortcuts.
5. When the task is completed, say "Next Action": "None".
6. Break tasks into subgoals and complete each one by one.
7. Avoid choosing the same action/elements multiple times in a row.
8. If prompted with login/captcha, say "Next Action": "None".
"""
        return main_section


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
