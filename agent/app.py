"""
OmniTool Agent - Gradio UI for AI-driven desktop control.

Usage:
    uv run app.py --desktops 192.168.1.10:5010 --parser-url 10.0.0.5:8013
    uv run app.py --desktops pc1:5010,pc2:5010 --parser-url gpu-server:8013 --ngrok
"""

import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import cast
import argparse
import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock
from loop import sampling_loop_sync
from tools import ToolResult
from tools import config as tools_config
import requests
from requests.exceptions import RequestException

INTRO_TEXT = '''
OmniParser + any LLM via **OpenRouter**. Enter your model name (e.g. `openai/gpt-4o`, `google/gemini-2.5-pro`, `anthropic/claude-sonnet-4`).

Check **Orchestrated** for multi-step planning with progress tracking. Type a message and press Send to start.
'''


def parse_arguments():
    parser = argparse.ArgumentParser(description="OmniTool Agent")
    parser.add_argument("--desktops", type=str, default="localhost:5010",
                        help="Comma-separated desktop server URLs (host:port)")
    parser.add_argument("--parser-url", type=str, default="localhost:8000",
                        help="Parser server URL (host:port)")
    parser.add_argument("--port", type=int, default=7888, help="Gradio UI port")
    parser.add_argument("--ngrok", action="store_true", help="Start ngrok tunnel")
    parser.add_argument("--ngrok-auth-token", type=str, default=None)
    return parser.parse_args()


args = parse_arguments()

# Parse desktop server URLs
desktop_urls = [url.strip() for url in args.desktops.split(",") if url.strip()]
tools_config.DESKTOP_SERVERS = desktop_urls
tools_config.DESKTOP_SERVER_URL = desktop_urls[0] if desktop_urls else "localhost:5010"
tools_config.ACTIVE_DESKTOP_INDEX = 0


class Sender:
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "model" not in state:
        state["model"] = "openai/gpt-4o"
    if "orchestrated" not in state:
        state["orchestrated"] = True
    if "api_key" not in state:
        state["api_key"] = os.getenv("OPENROUTER_API_KEY", "")
    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 2
    if "send_screenshots" not in state:
        state["send_screenshots"] = True
    if 'chatbot_messages' not in state:
        state['chatbot_messages'] = []
    if 'stop' not in state:
        state['stop'] = False


def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response


def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output


def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    def _render_message(message, hide_images=False):
        if isinstance(message, str):
            return message
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult) or message.__class__.__name__ == "ToolResult"
        )
        if not message or (is_tool_result and hide_images and not hasattr(message, "error") and not hasattr(message, "output")):
            return
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                return f'<img src="data:image/png;base64,{message.base64_image}">'
        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            return f"Analysis: {message.text}"
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            return f"Next action: {message.input}"
        else:
            return message

    message = _render_message(message, hide_images)
    if sender == "bot":
        chatbot_state.append({"role": "assistant", "content": message})
    else:
        chatbot_state.append({"role": "user", "content": message})


def valid_params(user_input, state):
    errors = []
    for server_name, url in [('Desktop', tools_config.DESKTOP_SERVER_URL), ('Parser', args.parser_url)]:
        try:
            probe_url = f'http://{url}/probe' if 'parser' not in server_name.lower() else f'http://{url}/probe/'
            response = requests.get(probe_url, timeout=3)
            if response.status_code != 200:
                errors.append(f"{server_name} server is not responding")
        except RequestException:
            errors.append(f"{server_name} server ({url}) is not responding")

    if not state["api_key"].strip():
        errors.append("OpenRouter API Key is not set")
    if not state["model"].strip():
        errors.append("Model name is not set")
    if not user_input:
        errors.append("No task provided")
    return errors


def process_input(user_input, state):
    if state["stop"]:
        state["stop"] = False

    errors = valid_params(user_input, state)
    if errors:
        raise gr.Error("Validation errors: " + ", ".join(errors))

    state["messages"].append({
        "role": Sender.USER,
        "content": [TextBlock(type="text", text=user_input)],
    })
    state['chatbot_messages'].append({"role": "user", "content": user_input})
    yield state['chatbot_messages']

    for loop_msg in sampling_loop_sync(
        model=state["model"],
        orchestrated=state["orchestrated"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=False),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        send_screenshots=state["send_screenshots"],
        max_tokens=16384,
        parser_url=args.parser_url,
    ):
        if loop_msg is None or state.get("stop"):
            yield state['chatbot_messages']
            break
        yield state['chatbot_messages']


def stop_app(state):
    state["stop"] = True
    return "Stopped"


# Check server connectivity on startup
print("\n--- Server Status ---")
for i, url in enumerate(desktop_urls):
    try:
        r = requests.get(f'http://{url}/probe', timeout=3)
        status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
    except Exception:
        status = "UNREACHABLE"
    marker = " (active)" if i == 0 else ""
    print(f"  Desktop {i+1}: {url} - {status}{marker}")

try:
    r = requests.get(f'http://{args.parser_url}/probe/', timeout=3)
    status = "OK" if r.status_code == 200 else f"HTTP {r.status_code}"
except Exception:
    status = "UNREACHABLE"
print(f"  Parser: {args.parser_url} - {status}")
print("---\n")


# Build Gradio UI
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.HTML("""<style>
        .markdown-text p { font-size: 18px; }
    </style>""")

    state = gr.State({})
    setup_state(state.value)

    gr.Markdown("# OmniTool")

    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(INTRO_TEXT, elem_classes="markdown-text")

    with gr.Accordion("Settings", open=True):
        with gr.Row():
            model = gr.Textbox(
                label="Model (OpenRouter)",
                value="openai/gpt-4o",
                placeholder="e.g. openai/gpt-4o, google/gemini-2.5-pro, anthropic/claude-sonnet-4",
                interactive=True,
                scale=3,
            )
            orchestrated = gr.Checkbox(
                label="Orchestrated",
                value=True,
                interactive=True,
                scale=1,
            )
            send_screenshots = gr.Checkbox(
                label="Send screenshots to LLM",
                value=True,
                interactive=True,
                scale=1,
            )
            only_n_images = gr.Slider(
                label="N most recent screenshots",
                minimum=0, maximum=10, step=1, value=2, interactive=True,
                scale=1,
            )
        with gr.Row():
            api_key = gr.Textbox(
                label="OpenRouter API Key", type="password",
                value=state.value.get("api_key", ""),
                placeholder="Paste your OpenRouter API key here",
                interactive=True,
            )

        if len(desktop_urls) > 1:
            desktop_dropdown = gr.Dropdown(
                label="Active Desktop",
                choices=desktop_urls,
                value=desktop_urls[0],
                interactive=True,
            )

            def update_active_desktop(selected_url, state):
                tools_config.DESKTOP_SERVER_URL = selected_url
                idx = desktop_urls.index(selected_url)
                tools_config.ACTIVE_DESKTOP_INDEX = idx
                print(f"Switched to desktop: {selected_url}")

            desktop_dropdown.change(fn=update_active_desktop, inputs=[desktop_dropdown, state], outputs=None)

    with gr.Row():
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a task for the agent...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")
        with gr.Column(scale=1, min_width=50):
            stop_button = gr.Button(value="Stop", variant="secondary")

    chatbot = gr.Chatbot(label="Agent History", autoscroll=True, height=580)

    def update_model(val, state):
        state["model"] = val

    def update_orchestrated(val, state):
        state["orchestrated"] = val

    def update_only_n_images(val, state):
        state["only_n_most_recent_images"] = val

    def update_send_screenshots(val, state):
        state["send_screenshots"] = val

    def update_api_key(val, state):
        state["api_key"] = val

    def clear_chat(state):
        state["messages"] = []
        state["responses"] = {}
        state["tools"] = {}
        state['chatbot_messages'] = []
        return state['chatbot_messages']

    model.change(fn=update_model, inputs=[model, state], outputs=None)
    orchestrated.change(fn=update_orchestrated, inputs=[orchestrated, state], outputs=None)
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    send_screenshots.change(fn=update_send_screenshots, inputs=[send_screenshots, state], outputs=None)
    api_key.change(fn=update_api_key, inputs=[api_key, state], outputs=None)
    chatbot.clear(fn=clear_chat, inputs=[state], outputs=[chatbot])

    submit_button.click(process_input, [chat_input, state], chatbot).then(lambda: "", outputs=chat_input)
    chat_input.submit(process_input, [chat_input, state], chatbot).then(lambda: "", outputs=chat_input)
    stop_button.click(stop_app, [state], None)

if __name__ == "__main__":
    if args.ngrok:
        from pyngrok import ngrok
        token = args.ngrok_auth_token or os.environ.get("NGROK_AUTHTOKEN")
        if token:
            ngrok.set_auth_token(token)
        tunnel = ngrok.connect(args.port, "http")
        print(f"Ngrok tunnel: {tunnel.public_url}")

    demo.launch(server_name="0.0.0.0", server_port=args.port)
