"""Agent sampling loop — routes to VLM or orchestrated VLM actor."""

from collections.abc import Callable

from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaContentBlock, BetaMessage, BetaMessageParam
from tools import ToolResult

from actors.llm.parser_client import ParserClient
from actors.vlm_actor import VLMAgent
from actors.vlm_orchestrated_actor import VLMOrchestratedAgent
from executor.executor import AnthropicExecutor


def sampling_loop_sync(
    *,
    model: str,
    orchestrated: bool,
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
    print(f'in sampling_loop_sync, model: {model}, orchestrated: {orchestrated}')
    parser_client = ParserClient(url=f"http://{parser_url}/parse/")

    if orchestrated:
        actor = VLMOrchestratedAgent(
            model=model, api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
            save_folder=save_folder,
        )
    else:
        actor = VLMAgent(
            model=model, api_key=api_key,
            api_response_callback=api_response_callback,
            output_callback=output_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
        )

    executor = AnthropicExecutor(
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
    )
    print(f"Model initialized: {model}")

    tool_result_content = None
    print(f"Starting message loop. Messages: {messages}")

    while True:
        parsed_screen = parser_client()
        tools_use_needed, vlm_response_json = actor(messages=messages, parsed_screen=parsed_screen)

        for message, tool_result_content in executor(tools_use_needed, messages):
            yield message

        if not tool_result_content:
            return messages
