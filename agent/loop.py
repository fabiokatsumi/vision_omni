"""Agent sampling loop that orchestrates the LLM actor and tool executor."""

from collections.abc import Callable
from enum import StrEnum

from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaContentBlock, BetaMessage, BetaMessageParam
from tools import ToolResult

from actors.llm.parser_client import ParserClient
from actors.anthropic_actor import AnthropicActor
from actors.vlm_actor import VLMAgent
from actors.vlm_orchestrated_actor import VLMOrchestratedAgent
from executor.executor import AnthropicExecutor

BETA_FLAG = "computer-use-2024-10-22"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
    APIProvider.OPENAI: "gpt-4o",
}


def sampling_loop_sync(
    *,
    model: str,
    provider: APIProvider | None,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = 2,
    max_tokens: int = 4096,
    parser_url: str,
    save_folder: str = "./uploads",
):
    """Synchronous agentic sampling loop."""
    print('in sampling_loop_sync, model:', model)
    parser_client = ParserClient(url=f"http://{parser_url}/parse/")

    if model == "claude-3-5-sonnet-20241022":
        actor = AnthropicActor(
            model=model, provider=provider, api_key=api_key,
            api_response_callback=api_response_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
        )
    elif model in {"omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl"}:
        actor = VLMAgent(
            model=model, provider=provider, api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
        )
    elif model in {"omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated", "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated"}:
        actor = VLMOrchestratedAgent(
            model=model, provider=provider, api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
            save_folder=save_folder,
        )
    else:
        raise ValueError(f"Model {model} not supported")

    executor = AnthropicExecutor(
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
    )
    print(f"Model initialized: {model}, Provider: {provider}")

    tool_result_content = None
    print(f"Starting message loop. Messages: {messages}")

    if model == "claude-3-5-sonnet-20241022":
        while True:
            parsed_screen = parser_client()
            screen_info_block = TextBlock(
                text='Below is the structured accessibility information of the current UI screen:\n' + parsed_screen['screen_info'],
                type='text',
            )
            messages.append({"role": "user", "content": [screen_info_block]})
            tools_use_needed = actor(messages=messages)

            for message, tool_result_content in executor(tools_use_needed, messages):
                yield message

            if not tool_result_content:
                return messages
            messages.append({"content": tool_result_content, "role": "user"})

    elif model in {"omniparser + gpt-4o", "omniparser + o1", "omniparser + o3-mini", "omniparser + R1", "omniparser + qwen2.5vl",
                    "omniparser + gpt-4o-orchestrated", "omniparser + o1-orchestrated", "omniparser + o3-mini-orchestrated",
                    "omniparser + R1-orchestrated", "omniparser + qwen2.5vl-orchestrated"}:
        while True:
            parsed_screen = parser_client()
            tools_use_needed, vlm_response_json = actor(messages=messages, parsed_screen=parsed_screen)

            for message, tool_result_content in executor(tools_use_needed, messages):
                yield message

            if not tool_result_content:
                return messages
