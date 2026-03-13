"""Anthropic Claude actor with native computer-use support."""

import asyncio
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import ToolResultBlockParam, TextBlock
from anthropic.types.beta import (
    BetaContentBlock, BetaContentBlockParam, BetaImageBlockParam,
    BetaMessage, BetaMessageParam, BetaTextBlockParam, BetaToolResultBlockParam,
    BetaTextBlock, BetaToolUseBlock,
)

from tools import ComputerTool, ToolCollection, ToolResult

BETA_FLAG = "computer-use-2024-10-22"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a computer system with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
"""


class AnthropicActor:
    def __init__(
        self,
        model: str,
        provider: APIProvider,
        api_key: str,
        api_response_callback: Callable[[APIResponse[BetaMessage]], None],
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        print_usage: bool = True,
    ):
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images

        self.tool_collection = ToolCollection(ComputerTool())
        self.system = SYSTEM_PROMPT
        self.total_token_usage = 0
        self.total_cost = 0
        self.print_usage = print_usage

        if provider == APIProvider.ANTHROPIC:
            self.client = Anthropic(api_key=api_key)
        elif provider == APIProvider.VERTEX:
            self.client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            self.client = AnthropicBedrock()

    def __call__(self, *, messages: list[BetaMessageParam]):
        if self.only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(messages, self.only_n_most_recent_images)

        raw_response = self.client.beta.messages.with_raw_response.create(
            max_tokens=self.max_tokens,
            messages=messages,
            model=self.model,
            system=self.system,
            tools=self.tool_collection.to_params(),
            betas=["computer-use-2024-10-22"],
        )

        self.api_response_callback(cast(APIResponse[BetaMessage], raw_response))
        response = raw_response.parse()

        self.total_token_usage += response.usage.input_tokens + response.usage.output_tokens
        self.total_cost += (response.usage.input_tokens * 3 / 1000000 + response.usage.output_tokens * 15 / 1000000)

        if self.print_usage:
            print(f"Claude total token usage: {self.total_token_usage}, cost: ${self.total_cost:.5f}")

        return response


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (message["content"] if isinstance(message["content"], list) else [])
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content
