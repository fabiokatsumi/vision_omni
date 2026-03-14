# OmniTool — Agent Architecture Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Servers](#3-servers)
   - [Desktop Server](#31-desktop-server)
   - [Parser Server](#32-parser-server)
   - [Agent UI Server](#33-agent-ui-server)
4. [Agent Loop & Actors](#4-agent-loop--actors)
   - [Sampling Loop](#41-sampling-loop)
   - [VLMAgent](#42-vlmagent--single-step-mode)
   - [VLMOrchestratedAgent](#43-vlmorchestrated-agent--multi-step-mode)
   - [AnthropicActor](#44-anthropicactor--native-claude-computer-use)
5. [Tools](#5-tools)
   - [ComputerTool](#51-computertool)
   - [ToolCollection](#52-toolcollection)
   - [Tool Execution Flow](#53-tool-execution-flow)
6. [Prompts](#6-prompts)
   - [VLM System Prompt](#61-vlm-system-prompt)
   - [Orchestrator Ledger Prompt](#62-orchestrator-ledger-prompt)
   - [Task Planning Prompt](#63-task-planning-prompt)
   - [Anthropic System Prompt](#64-anthropic-system-prompt)
7. [Dependencies](#7-dependencies)
8. [Environment Variables](#8-environment-variables)
9. [Suggested Improvements](#9-suggested-improvements)

---

## 1. Overview

OmniTool is an AI-driven desktop automation agent built on top of Microsoft's [OmniParser](https://github.com/microsoft/OmniParser) research. It enables any vision-language model (VLM) — routed through OpenRouter — to control a remote or local desktop computer by interpreting screenshots, identifying UI elements, and executing mouse/keyboard actions.

The system is composed of **three independent servers** that communicate over HTTP:

| Server | Framework | Purpose |
|--------|-----------|---------|
| **Desktop Server** | Flask | Runs on the PC being controlled. Captures screenshots and executes mouse/keyboard actions via PyAutoGUI. |
| **Parser Server** | FastAPI | Runs on a GPU machine. Processes screenshots using YOLO object detection, OCR (EasyOCR/PaddleOCR), and Florence-2 captioning to identify all UI elements. |
| **Agent UI** | Gradio | Orchestrator with web interface. Routes the agent loop: captures screen → parses it → sends to LLM → executes returned actions. |

The agent loop works as follows:
1. Capture a screenshot from the Desktop Server
2. Send it to the Parser Server, which returns bounding boxes, labels, and an annotated SOM (Set-of-Mark) image
3. Pass the parsed screen info + SOM image to a VLM via OpenRouter
4. The VLM responds with a JSON action (e.g., "left_click on Box ID 5")
5. The Executor translates this into a PyAutoGUI action and sends it to the Desktop Server
6. Repeat until the task is complete or the VLM returns `"None"`

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agent UI (Gradio)                        │
│                     agent/app.py — port 7888                    │
│                                                                  │
│  ┌─────────────┐  ┌───────────────┐  ┌────────────────────────┐ │
│  │ Settings UI  │  │ Chat Display  │  │ State Management       │ │
│  │ - Model      │  │ - SOM images  │  │ - messages             │ │
│  │ - API Key    │  │ - Reasoning   │  │ - model config         │ │
│  │ - Orch. mode │  │ - Actions     │  │ - tool results         │ │
│  │ - Desktop    │  │ - Latency     │  │ - chatbot history      │ │
│  └─────────────┘  └───────────────┘  └────────────────────────┘ │
│                            │                                     │
│                   sampling_loop_sync()                           │
│                     agent/loop.py                                │
│                            │                                     │
│              ┌─────────────┼─────────────┐                      │
│              ▼             ▼             ▼                       │
│      ┌──────────┐  ┌────────────┐  ┌──────────────┐            │
│      │  Parser  │  │   Actor    │  │   Executor   │            │
│      │  Client  │  │ VLM/Orch.  │  │ (tool exec)  │            │
│      └─────┬────┘  └─────┬──────┘  └──────┬───────┘            │
└────────────┼─────────────┼────────────────┼─────────────────────┘
             │             │                │
             ▼             ▼                ▼
   ┌─────────────────┐  ┌──────────┐  ┌────────────────────┐
   │  Parser Server   │  │OpenRouter│  │  Desktop Server     │
   │  FastAPI — :8000  │  │   API    │  │  Flask — :5010      │
   │                   │  │(external)│  │                     │
   │ YOLO + OCR +      │  └──────────┘  │ PyAutoGUI actions   │
   │ Florence-2        │                │ Screenshot capture   │
   └───────────────────┘                └─────────────────────┘
```

**Communication protocol:** All inter-server communication is HTTP/REST with JSON payloads. Images are transferred as base64-encoded strings.

---

## 3. Servers

### 3.1 Desktop Server

**File:** `desktop/server.py`
**Framework:** Flask
**Default port:** 5010
**Purpose:** Runs on the target PC to provide remote control capabilities.

#### Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| `GET` | `/probe` | Health check | — | `{"status": "ok", "message": "Desktop server is running"}` |
| `GET` | `/screenshot` | Capture the current screen | — | PNG image (binary), with optional cursor overlay |
| `GET` | `/screen_size` | Get display dimensions | — | `{"width": int, "height": int}` |
| `GET` | `/mouse_position` | Get current cursor position | — | `{"x": int, "y": int}` |
| `POST` | `/action` | Execute a PyAutoGUI action | `{"action": str, "args": [], "kwargs": {}}` | `{"status": "success", "result": ...}` |

#### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `5010` | Port to listen on |
| `--ngrok` | off | Start ngrok tunnel for remote access |
| `--ngrok-auth-token` | — | Ngrok auth token (or `NGROK_AUTHTOKEN` env var) |
| `--log-file` | stdout | Log output path |

#### Key Features

- **Thread locking:** All endpoints use a `threading.Lock()` (`control_lock`) to serialize GUI operations and prevent race conditions.
- **Cursor overlay:** If `cursor.png` exists in the server directory, screenshots include a cursor image pasted at the current mouse position.
- **PyAutoGUI safety disabled:** `pyautogui.FAILSAFE = False` — the fail-safe corner feature is turned off for automation reliability.
- **Self-test on startup:** Validates screen access, mouse position, and screenshot capture before starting the server.
- **Action execution:** The `/action` endpoint dynamically calls any `pyautogui` function by name (e.g., `click`, `moveTo`, `typewrite`, `press`, `scroll`, `keyDown`, `keyUp`, `dragTo`).

---

### 3.2 Parser Server

**File:** `parser/server.py`
**Framework:** FastAPI + Uvicorn
**Default port:** 8000
**Purpose:** Processes screenshots to detect and label all UI elements using computer vision models.

#### Endpoints

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| `POST` | `/parse/` | Parse a screenshot | `{"base64_image": str}` | `{"som_image_base64": str, "parsed_content_list": list, "latency": float}` |
| `GET` | `/probe/` | Health check | — | `{"status": "ok", "message": "Parser server is running"}` |

#### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--som-model-path` | `../weights/icon_detect/model.pt` | Path to YOLO detection model |
| `--caption-model-name` | `florence2` | Caption model name |
| `--caption-model-path` | `../weights/icon_caption_florence` | Path to caption model directory |
| `--device` | `cpu` | Compute device (`cuda` or `cpu`) |
| `--box-threshold` | `0.05` | YOLO detection confidence threshold |
| `--weights-dir` | `../weights` | Root directory for model weights |
| `--host` | `0.0.0.0` | Host to bind |
| `--port` | `8000` | Port to listen on |
| `--ngrok` | off | Start ngrok tunnel |
| `--ngrok-auth-token` | — | Ngrok auth token |

#### Vision Pipeline (OmniParser)

The parser uses the `Omniparser` class (`parser/omniparser.py`) which chains three models:

1. **YOLO icon detection** (`ultralytics==8.3.70`): Detects UI element bounding boxes (icons, buttons, etc.) using a fine-tuned YOLO model from `microsoft/OmniParser-v2.0`.
2. **OCR text detection** (`easyocr` + `paddleocr`): Extracts text content from the screenshot. EasyOCR is used by default with `text_threshold=0.8`.
3. **Florence-2 captioning** (`transformers`): Generates semantic descriptions for detected icons (e.g., "search button", "settings gear icon").

**Output format:** Each detected element is returned as:
```json
{
    "type": "text" | "icon",
    "content": "description or text content",
    "bbox": [x1, y1, x2, y2]  // normalized coordinates (0-1)
}
```

The SOM (Set-of-Mark) image is the original screenshot with numbered bounding boxes overlaid on every detected element.

#### Key Features

- **Auto-download weights:** `setup_weights.py` automatically downloads model weights from HuggingFace (`microsoft/OmniParser-v2.0`) if not present.
- **Self-test on startup:** Parses a synthetic 200x200 white image to verify models loaded correctly.
- **Dynamic box scaling:** Annotation sizes (text, borders) scale with image resolution using `box_overlay_ratio = max(image.size) / 3200`.

---

### 3.3 Agent UI Server

**File:** `agent/app.py`
**Framework:** Gradio
**Default port:** 7888
**Purpose:** Web-based orchestrator UI that ties everything together.

#### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--desktops` | `localhost:5010` | Comma-separated desktop server URLs (e.g., `pc1:5010,pc2:5010`) |
| `--parser-url` | `localhost:8000` | Parser server URL |
| `--port` | `7888` | Gradio UI port |
| `--ngrok` | off | Start ngrok tunnel |
| `--ngrok-auth-token` | — | Ngrok auth token |

#### UI Components

- **Model field:** OpenRouter model name (e.g., `openai/gpt-4o`, `google/gemini-2.5-pro`, `anthropic/claude-sonnet-4`)
- **API Key field:** OpenRouter API key (password-masked)
- **Orchestrated checkbox:** Toggle multi-step planning mode (default: on)
- **N most recent screenshots slider:** Controls how many historical screenshots to keep in context (0-10, default: 2)
- **Desktop dropdown:** Appears when multiple desktops are configured; allows runtime switching
- **Chat input:** Task description for the agent
- **Send / Stop buttons:** Start or interrupt the agent loop
- **Chat history:** Displays SOM images, reasoning, actions, latency, and parsed elements

#### State Management

The Gradio app maintains per-session state:
```python
{
    "messages": [],              # Conversation history (Anthropic format)
    "model": "openai/gpt-4o",   # Selected model
    "orchestrated": True,        # Multi-step mode
    "api_key": "",               # OpenRouter API key
    "auth_validated": False,     # Validation flag
    "responses": {},             # API response log
    "tools": {},                 # Tool result log
    "only_n_most_recent_images": 2,  # Image context limit
    "chatbot_messages": [],      # Gradio chat display
    "stop": False                # Stop signal
}
```

#### Startup Validation

On launch, the agent probes all configured servers:
```
--- Server Status ---
  Desktop 1: localhost:5010 - OK (active)
  Desktop 2: pc2:5010 - UNREACHABLE
  Parser: gpu-server:8013 - OK
---
```

---

## 4. Agent Loop & Actors

### 4.1 Sampling Loop

**File:** `agent/loop.py`
**Function:** `sampling_loop_sync()`

This is the core orchestration function. It runs a synchronous loop:

```
┌─► ParserClient() — capture screenshot, send to parser, get SOM + elements
│       │
│       ▼
│   Actor() — send parsed screen to VLM, get action response
│       │
│       ▼
│   Executor() — execute tool actions (mouse/keyboard via Desktop Server)
│       │
│       ▼
│   tool_result_content? ──── No ──► EXIT (task complete)
│       │
│      Yes
└───────┘
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | — | OpenRouter model name |
| `orchestrated` | `bool` | — | Enable multi-step planning |
| `messages` | `list` | — | Conversation history |
| `output_callback` | `Callable` | — | UI display callback |
| `tool_output_callback` | `Callable` | — | Tool result callback |
| `api_response_callback` | `Callable` | — | API response callback |
| `api_key` | `str` | — | OpenRouter API key |
| `only_n_most_recent_images` | `int` | `2` | Keep N most recent screenshots |
| `max_tokens` | `int` | `4096` | Max LLM output tokens |
| `parser_url` | `str` | — | Parser server URL |
| `save_folder` | `str` | `./uploads` | Trajectory save path (orchestrated mode) |

**Termination conditions:**
- The actor returns `"Next Action": "None"` (task complete or blocked)
- The executor returns no `tool_result_content` (no tools were called)
- The user clicks "Stop" in the UI

---

### 4.2 VLMAgent — Single-Step Mode

**File:** `agent/actors/vlm_actor.py`
**Class:** `VLMAgent`

The simplest actor. Each iteration:
1. Receives the parsed screen (SOM image, element list, bounding boxes)
2. Constructs a system prompt with all detected elements
3. Appends screenshot + SOM image to the message history
4. Calls the VLM via OpenRouter
5. Parses the JSON response to extract the action
6. Converts the action into Anthropic `BetaToolUseBlock` objects

**Action resolution:**
- The VLM returns a `"Box ID"` — the actor looks up the bounding box coordinates in `parsed_content_list`
- Computes the centroid: `x = (x1 + x2) / 2 * screen_width`, `y = (y1 + y2) / 2 * screen_height`
- Creates a `mouse_move` tool block to move to the centroid, then the actual action block (click, type, etc.)
- Draws a red circle on the SOM image at the target location for visual feedback

**Image management:**
- SOM images from previous steps are removed from history (`_remove_som_images()`)
- Old images beyond `only_n_most_recent_images` are filtered out (`_maybe_filter_to_n_most_recent_images()`)

---

### 4.3 VLMOrchestratedAgent — Multi-Step Mode

**File:** `agent/actors/vlm_orchestrated_actor.py`
**Class:** `VLMOrchestratedAgent`

Extends the VLMAgent with task planning and progress tracking:

#### Step 0 — Task Planning
On the first call, before any actions:
1. Sends the user's task to the VLM with a planning prompt
2. The VLM returns a bullet-point plan in JSON format
3. The plan is saved to `{save_folder}/plan.json` and displayed in the UI

#### Steps 1+ — Ledger Check + Action
On each subsequent call:
1. Sends the conversation history + the orchestrator ledger prompt to the VLM
2. The VLM evaluates progress:
   - Is the request fully satisfied?
   - Are we in a loop?
   - Are we making forward progress?
   - What instruction should we give next?
3. The ledger is displayed as a collapsible section in the UI
4. Then proceeds with the normal VLMAgent flow (parse screen → get action)

#### Trajectory Saving
Each step saves to `{save_folder}/trajectory.json` (JSONL format):
```json
{
    "screenshot_path": "path/to/screenshot_N.png",
    "som_screenshot_path": "path/to/som_screenshot_N.png",
    "screen_info": "ID: 0, Text: ...\nID: 1, Icon: ...",
    "latency_omniparser": 1.23,
    "latency_vlm": 2.45,
    "vlm_response_json": {"Reasoning": "...", "Next Action": "..."},
    "ledger": "{...}"
}
```

---

### 4.4 AnthropicActor — Native Claude Computer-Use

**File:** `agent/actors/anthropic_actor.py`
**Class:** `AnthropicActor`

An alternative actor that uses Anthropic's native computer-use API (beta flag `computer-use-2024-10-22`) instead of OpenRouter. This actor:

- Calls the Anthropic API directly with `tools=self.tool_collection.to_params()`
- Supports three providers: `Anthropic`, `AnthropicBedrock`, `AnthropicVertex`
- Tracks token usage and cost (input: $3/1M tokens, output: $15/1M tokens)
- Has its own image filtering logic

**Current status:** This actor is **not integrated** into the main `sampling_loop_sync()`. It exists as standalone code that could be used for direct Claude computer-use without the OmniParser pipeline.

---

## 5. Tools

### 5.1 ComputerTool

**File:** `agent/tools/computer.py`
**Class:** `ComputerTool`

The primary tool that translates agent actions into Desktop Server HTTP requests.

#### Available Actions

| Action | Parameters | Description | PyAutoGUI Call |
|--------|-----------|-------------|----------------|
| `mouse_move` | `coordinate: [x, y]` | Move cursor to position | `moveTo(x, y)` |
| `left_click` | — | Click at current position | `click()` |
| `right_click` | — | Right-click at current position | `rightClick()` |
| `double_click` | — | Double-click at current position | `doubleClick()` |
| `middle_click` | — | Middle-click at current position | `middleClick()` |
| `left_click_drag` | `coordinate: [x, y]` | Drag from current to target position | `dragTo(x, y, duration=0.5)` |
| `key` | `text: str` | Press key combination (e.g., `ctrl+c`) | `keyDown()` / `keyUp()` per key |
| `type` | `text: str` | Type text and press Enter | `click()` → `typewrite(text)` → `press("enter")` |
| `screenshot` | — | Capture current screen | GET `/screenshot` |
| `cursor_position` | — | Get current cursor coordinates | GET `/mouse_position` |
| `scroll_up` | — | Scroll up 100 units | `scroll(100)` |
| `scroll_down` | — | Scroll down 100 units | `scroll(-100)` |
| `hover` | — | No-op (cursor already moved) | — |
| `wait` | — | Wait 1 second | `time.sleep(1)` |

#### Key Conversion Map

Special key names are translated before being sent to PyAutoGUI:

| Agent Key | PyAutoGUI Key |
|-----------|---------------|
| `Page_Down` | `pagedown` |
| `Page_Up` | `pageup` |
| `Super_L` | `win` |
| `Escape` | `esc` |

#### Coordinate Scaling

The tool handles resolution differences between the actual screen and the VLM's input resolution:

- **Target resolutions:** XGA (1024x768), WXGA (1280x800), FWXGA (1366x768)
- **Scaling logic:** Matches the target resolution with the closest aspect ratio to the actual screen
- **Default:** WXGA (1280x800)
- **Direction:** `API → Computer` scales up; `Computer → API` scales down

#### HTTP Communication

All actions are sent to the Desktop Server via `POST /action`:
```json
{"action": "click", "args": [], "kwargs": {}}
{"action": "moveTo", "args": [500, 300], "kwargs": {}}
{"action": "typewrite", "args": ["hello world"], "kwargs": {"interval": 0.012}}
```

A 0.7-second delay is added after each action call to allow the UI to update.

---

### 5.2 ToolCollection

**File:** `agent/tools/collection.py`

A lightweight registry that maps tool names to tool instances:
- `run(name, tool_input)` — routes execution to the matching tool
- `to_params()` — returns tool definitions in Anthropic API format

Currently only contains one tool: `ComputerTool`.

---

### 5.3 Tool Execution Flow

**File:** `agent/executor/executor.py`
**Class:** `AnthropicExecutor`

The executor processes the actor's response:

```
BetaMessage from Actor
    │
    ├─► BetaTextBlock → output_callback() (display reasoning)
    │
    └─► BetaToolUseBlock → ToolCollection.run()
            │
            ├─► ComputerTool.__call__(action, text, coordinate)
            │       │
            │       └─► HTTP POST to Desktop Server /action
            │
            └─► ToolResult → _make_api_tool_result()
                    │
                    └─► BetaToolResultBlockParam (appended to messages)
```

Tool results are formatted back into Anthropic's message format with text output, error messages, or base64 screenshot images.

---

## 6. Prompts

### 6.1 VLM System Prompt

**Used by:** `VLMAgent._get_system_prompt()` and `VLMOrchestratedAgent._get_system_prompt()`
**File:** `agent/actors/vlm_actor.py:154-197`

This prompt defines the agent's capabilities and expected output format:

```
You are using a computer.
You are able to use a mouse and keyboard to interact with the computer
based on the given task and screenshot.
You can only interact with the desktop GUI (no terminal or application
menu access).

Here is the list of all detected bounding boxes by IDs on the screen
and their description:
  ID: 0, Text: File
  ID: 1, Icon: search button
  ...

Your available "Next Action" only include:
- type: types a string of text.
- left_click: move mouse to box id and left clicks.
- right_click: move mouse to box id and right clicks.
- double_click: move mouse to box id and double clicks.
- hover: move mouse to box id.
- scroll_up: scrolls the screen up.
- scroll_down: scrolls the screen down.
- wait: waits for 1 second.

Output format:
{
    "Reasoning": str,
    "Next Action": "action_type, description" | "None",
    "Box ID": n,
    "value": "xxx"
}

IMPORTANT NOTES:
1. Single action at a time.
2. Analyze current screen and reflect on history.
3. When done: "Next Action": "None".
4. No keyboard shortcuts.
5. Break tasks into subgoals.
6. Avoid repeating same action.
7. If login/captcha: "Next Action": "None".
```

### 6.2 Orchestrator Ledger Prompt

**Used by:** `VLMOrchestratedAgent._update_ledger()`
**File:** `agent/actors/vlm_orchestrated_actor.py:23-55`

Sent to the VLM before each action step in orchestrated mode:

```
Recall we are working on the following request:
  {task}

To make progress on the request, please answer:
- Is the request fully satisfied? (True/False)
- Are we in a loop repeating the same requests?
- Are we making forward progress?
- What instruction or question would you give to complete the task?

Output JSON:
{
    "is_request_satisfied": {"reason": str, "answer": bool},
    "is_in_loop": {"reason": str, "answer": bool},
    "is_progress_being_made": {"reason": str, "answer": bool},
    "instruction_or_question": {"reason": str, "answer": str}
}
```

### 6.3 Task Planning Prompt

**Used by:** `VLMOrchestratedAgent._initialize_task()`
**File:** `agent/actors/vlm_orchestrated_actor.py:270-281`

Sent once at the start of an orchestrated session:

```
Please devise a short bullet-point plan for addressing the original
user task: {task}

Write your plan in JSON:
{
    "step 1": "...",
    "step 2": "...",
    ...
}
```

### 6.4 Anthropic System Prompt

**Used by:** `AnthropicActor`
**File:** `agent/actors/anthropic_actor.py:29-33`

A minimal capability declaration for Claude's native computer-use mode:

```
<SYSTEM_CAPABILITY>
* You are utilizing a computer system with internet access.
* The current date is [today's date].
</SYSTEM_CAPABILITY>
```

---

## 7. Dependencies

### Agent Module (`agent/pyproject.toml`)

| Package | Version | Purpose |
|---------|---------|---------|
| `gradio` | >=6.9.0 | Web UI framework |
| `anthropic` | >=0.84.0 | Anthropic SDK (message types, API client) |
| `requests` | >=2.32.5 | HTTP client for server communication |
| `pillow` | >=12.1.1 | Image processing (resize, annotate) |
| `pyngrok` | — | Optional ngrok tunnel support |

**Python:** >=3.12, <3.13

### Parser Module (`parser/pyproject.toml`)

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | >=0.115.0 | Web framework |
| `uvicorn` | >=0.34.0 | ASGI server |
| `pydantic` | — | Request/response validation |
| `torch` | — | PyTorch deep learning |
| `torchvision` | — | Vision utilities |
| `numpy` | ==1.26.4 | Numerical computing |
| `opencv-python` | — | Image processing |
| `easyocr` | — | OCR engine (English) |
| `paddlepaddle` | — | PaddlePaddle framework |
| `paddleocr` | >=2.7, <3 | PaddleOCR text detection |
| `supervision` | ==0.18.0 | Object detection utilities |
| `ultralytics` | ==8.3.70 | YOLO model framework |
| `transformers` | >=4.38, <4.46 | HuggingFace transformers (Florence-2) |
| `accelerate` | — | Multi-GPU acceleration |
| `timm` | — | PyTorch image models |
| `einops` | ==0.8.0 | Tensor manipulation |
| `pillow` | >=12.1.1 | Image processing |
| `huggingface-hub` | — | Model weight downloading |
| `pyngrok` | — | Optional ngrok tunnel |

**Python:** >=3.12, <3.13

### Desktop Module (`desktop/pyproject.toml`)

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | — | Web framework |
| `pyautogui` | — | Desktop automation (mouse/keyboard) |
| `pillow` | — | Image processing |
| `pyngrok` | — | Optional ngrok tunnel |

**Python:** >=3.10

### Model Weights (auto-downloaded)

From HuggingFace `microsoft/OmniParser-v2.0`:
- `icon_detect/model.pt` — YOLO icon detection model
- `icon_caption_florence/` — Florence-2 caption model

---

## 8. Environment Variables

| Variable | Module | Required | Description |
|----------|--------|----------|-------------|
| `OPENROUTER_API_KEY` | Agent | Yes (or enter in UI) | OpenRouter API authentication |
| `NGROK_AUTHTOKEN` | All | No | Ngrok tunnel authentication |
| `HIDE_WARNING` | Agent | No | Hide intro text in Gradio UI |

---

## 9. Suggested Improvements

### 9.1 Error Handling & Resilience

- **Bare `except` clauses:** `vlm_actor.py:101` and `vlm_orchestrated_actor.py:159` silently catch all exceptions when processing bounding boxes. These should catch specific exceptions and log warnings.
- **No retry logic:** The OpenRouter client (`openrouter_client.py`) has no retry/backoff for transient API failures. Consider using `tenacity` or `urllib3.util.retry`.
- **JSON parse failures:** The VLM response is parsed with `json.loads()` without validation. If the model returns malformed JSON, the agent crashes. Add a JSON schema validator or fallback parser.
- **Desktop Server action injection:** The `/action` endpoint calls `getattr(pyautogui, action_name)` with any string, which could execute unintended functions. Add an allowlist of permitted actions.

### 9.2 Testing

- **No tests exist.** The project has zero test files, no pytest configuration, and no CI/CD pipeline. Priority additions:
  - Unit tests for `ComputerTool` action routing and coordinate scaling
  - Unit tests for `VLMAgent` JSON response parsing
  - Integration tests for the parser pipeline (YOLO + OCR + Florence-2)
  - End-to-end tests with mocked Desktop/Parser servers
  - Add pytest + pytest-asyncio to dev dependencies

### 9.3 AnthropicActor Integration

- The `AnthropicActor` class is fully implemented but **not wired into the main loop**. It could be integrated as a third actor option (alongside VLMAgent and VLMOrchestratedAgent) to support Claude's native computer-use without the OmniParser pipeline, or with a hybrid approach using OmniParser for element detection and Claude for action generation.

### 9.4 Containerization

- **No Docker setup.** The three-server architecture is ideal for Docker Compose:
  ```yaml
  services:
    desktop:
      build: ./desktop
      ports: ["5010:5010"]
    parser:
      build: ./parser
      deploy:
        resources:
          reservations:
            devices:
              - capabilities: [gpu]
      ports: ["8000:8000"]
    agent:
      build: ./agent
      ports: ["7888:7888"]
      depends_on: [desktop, parser]
  ```

### 9.5 Structured Logging

- Current logging is inconsistent: some modules use `print()`, others use `logging`. Standardize on Python's `logging` module with structured output (JSON format) for production observability.
- Add request IDs to trace actions across the three servers.

### 9.6 Memory & Context Management

- **Image accumulation:** Screenshots are saved to `./tmp/outputs/` without cleanup. Add a retention policy or cleanup on session end.
- **Message history growth:** The `only_n_most_recent_images` filter helps, but text messages accumulate indefinitely. Consider summarizing old conversation history or implementing a sliding window.
- **Token limits:** No guard against exceeding the model's context window. Add token counting before API calls and truncate history if needed.

### 9.7 Security

- **No authentication** on any server. Anyone with network access can control the desktop or invoke the parser. Add API key authentication or mutual TLS.
- **Action injection risk:** The Desktop Server's `/action` endpoint accepts any PyAutoGUI function name. Restrict to an explicit allowlist.
- **API key exposure:** The OpenRouter key is stored in Gradio session state (in-memory), but could be logged in error messages. Mask keys in logs.

### 9.8 Performance

- **Synchronous blocking:** The agent loop is fully synchronous. The parser call and VLM call could run concurrently (parser prepares next screen while VLM processes current one).
- **Fixed delays:** `ComputerTool` adds 0.7s after every action and 2.0s after screenshots. These could be adaptive based on system responsiveness.
- **Image compression:** Screenshots are transferred as full PNG. Using JPEG with quality 85 could reduce transfer size by 60-80%.

### 9.9 Multi-Model & Provider Support

- **OpenRouter-only:** The VLM actors are hardcoded to OpenRouter. Consider abstracting the LLM client to support direct API calls to providers (OpenAI, Google, Anthropic) without the OpenRouter middleman.
- **Model-specific prompting:** Different VLMs may respond better to different prompt formats. The current one-size-fits-all system prompt could be customized per model family.

### 9.10 Agent Capabilities

- **No keyboard shortcuts:** The system prompt explicitly forbids keyboard shortcuts (`Ctrl+C`, `Alt+Tab`, etc.). This limits the agent's efficiency for many common tasks. Consider adding keyboard shortcut support with a safety allowlist.
- **No drag-and-drop support in VLM prompts:** While `left_click_drag` exists in the tool, the VLM system prompt does not list it as an available action.
- **No text selection:** The agent cannot select text (shift+click, triple-click for line). Adding this would enable copy-paste workflows.
- **No multi-monitor support:** The Desktop Server captures the primary screen only. PyAutoGUI supports multi-monitor; this could be exposed.

### 9.11 Observability & Debugging

- **Trajectory is only saved in orchestrated mode.** Save action histories in single-step mode too for debugging.
- **No metrics collection.** Add Prometheus/StatsD metrics for: actions per minute, parser latency, VLM latency, error rates, token usage.
- **No session replay.** The saved trajectories could power a replay UI for debugging failed automation runs.
