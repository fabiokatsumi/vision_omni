import base64
import time
from enum import StrEnum
from typing import Literal, TypedDict

from PIL import Image

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .screen_capture import get_screenshot
from . import config as tools_config
import requests

OUTPUT_DIR = "./tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key", "type", "mouse_move", "left_click", "left_click_drag",
    "right_click", "middle_click", "double_click", "screenshot",
    "cursor_position", "hover", "wait"
]


class Resolution(TypedDict):
    width: int
    height: int


MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),
    "WXGA": Resolution(width=1280, height=800),
    "FWXGA": Resolution(width=1366, height=768),
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self, is_scaling: bool = False):
        super().__init__()
        self.display_num = None
        self.offset_x = 0
        self.offset_y = 0
        self.is_scaling = is_scaling
        self.width, self.height = self.get_screen_size()
        print(f"screen size: {self.width}, {self.height}")
        self.key_conversion = {
            "Page_Down": "pagedown", "Page_Up": "pageup",
            "Super_L": "win", "Escape": "esc",
        }

    async def __call__(self, *, action: Action, text: str | None = None, coordinate: tuple[int, int] | None = None, **kwargs):
        print(f"action: {action}, text: {text}, coordinate: {coordinate}, is_scaling: {self.is_scaling}")
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            if self.is_scaling:
                x, y = self.scale_coordinates(ScalingSource.API, coordinate[0], coordinate[1])
            else:
                x, y = coordinate

            if action == "mouse_move":
                self.call_action("moveTo", args=[x, y])
                return ToolResult(output=f"Moved mouse to ({x}, {y})")
            elif action == "left_click_drag":
                pos = self.get_mouse_position()
                current_x, current_y = pos["x"], pos["y"]
                self.call_action("dragTo", args=[x, y], kwargs={"duration": 0.5})
                return ToolResult(output=f"Dragged mouse from ({current_x}, {current_y}) to ({x}, {y})")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                keys = text.split('+')
                for key in keys:
                    key = self.key_conversion.get(key.strip(), key.strip()).lower()
                    self.call_action("keyDown", args=[key])
                for key in reversed(keys):
                    key = self.key_conversion.get(key.strip(), key.strip()).lower()
                    self.call_action("keyUp", args=[key])
                return ToolResult(output=f"Pressed keys: {text}")
            elif action == "type":
                self.call_action("click")
                self.call_action("typewrite", args=[text], kwargs={"interval": TYPING_DELAY_MS / 1000})
                self.call_action("press", args=["enter"])
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(output=text, base64_image=screenshot_base64)

        if action in ("left_click", "right_click", "double_click", "middle_click", "screenshot", "cursor_position", "left_press"):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                pos = self.get_mouse_position()
                x, y = self.scale_coordinates(ScalingSource.COMPUTER, pos["x"], pos["y"])
                return ToolResult(output=f"X={x},Y={y}")
            else:
                action_map = {
                    "left_click": "click", "right_click": "rightClick",
                    "middle_click": "middleClick", "double_click": "doubleClick",
                }
                if action == "left_press":
                    self.call_action("mouseDown")
                    time.sleep(1)
                    self.call_action("mouseUp")
                else:
                    self.call_action(action_map[action])
                return ToolResult(output=f"Performed {action}")

        if action in ("scroll_up", "scroll_down"):
            scroll_val = 100 if action == "scroll_up" else -100
            self.call_action("scroll", args=[scroll_val])
            return ToolResult(output=f"Performed {action}")
        if action == "hover":
            return ToolResult(output=f"Performed {action}")
        if action == "wait":
            time.sleep(1)
            return ToolResult(output=f"Performed {action}")
        raise ToolError(f"Invalid action: {action}")

    def _base_url(self):
        return f"http://{tools_config.DESKTOP_SERVER_URL}"

    def call_action(self, action_name: str, args: list | None = None, kwargs: dict | None = None):
        payload = {"action": action_name}
        if args:
            payload["args"] = args
        if kwargs:
            payload["kwargs"] = kwargs

        try:
            print(f"calling action: {action_name} args={args} kwargs={kwargs}")
            response = requests.post(
                f"{self._base_url()}/action",
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=90,
            )
            time.sleep(0.7)
            if response.status_code != 200:
                raise ToolError(f"Failed to execute action {action_name}. Status code: {response.status_code}")
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ToolError(f"An error occurred while trying to execute action {action_name}: {str(e)}")

    def get_mouse_position(self) -> dict:
        try:
            response = requests.get(f"{self._base_url()}/mouse_position", timeout=90)
            if response.status_code != 200:
                raise ToolError(f"Failed to get mouse position. Status code: {response.status_code}")
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ToolError(f"An error occurred while trying to get mouse position: {str(e)}")

    async def screenshot(self):
        if not hasattr(self, 'target_dimension'):
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]
        width, height = self.target_dimension["width"], self.target_dimension["height"]
        screenshot, path = get_screenshot(resize=True, target_width=width, target_height=height)
        time.sleep(0.7)
        return ToolResult(base64_image=base64.b64encode(path.read_bytes()).decode())

    def padding_image(self, screenshot):
        _, height = screenshot.size
        new_width = height * 16 // 10
        padding_image = Image.new("RGB", (new_width, height), (255, 255, 255))
        padding_image.paste(screenshot, (0, 0))
        return padding_image

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None

        for target_name, dimension in MAX_SCALING_TARGETS.items():
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                    self.target_dimension = target_dimension
                break

        if target_dimension is None:
            target_dimension = MAX_SCALING_TARGETS["WXGA"]
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]

        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        return round(x * x_scaling_factor), round(y * y_scaling_factor)

    def get_screen_size(self):
        try:
            response = requests.get(f"{self._base_url()}/screen_size", timeout=90)
            if response.status_code != 200:
                raise ToolError(f"Failed to get screen size. Status code: {response.status_code}")
            data = response.json()
            return data["width"], data["height"]
        except requests.exceptions.RequestException as e:
            raise ToolError(f"An error occurred while trying to get screen size: {str(e)}")
