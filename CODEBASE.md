# OmniParser / OmniTool - Codebase Documentation

## Project Overview

**OmniParser** is a vision-based GUI parsing system that converts screenshots into structured, machine-readable UI elements. It combines YOLO-based icon detection, OCR (EasyOCR/PaddleOCR), and Florence-2 caption generation to produce annotated screen representations that AI agents can understand and act upon.

**OmniTool** is the integrated platform built on top of OmniParser. It connects a Gradio-based UI, multiple LLM/VLM providers, and a Windows 11 VM to enable autonomous computer-use agents that can navigate and interact with desktop applications.

---

## Architecture

The system is composed of three main components:

```
+---------------------+       +----------------------+       +---------------------+
|   Gradio UI + Agent |       |  OmniParser Server   |       |   OmniBox (VM)      |
|   Loop (port 7888)  |<----->|  FastAPI (port 8000)  |       |  Windows 11 + Flask |
|                     |       |  YOLO + Florence-2   |       |  (port 5006/8006)   |
|  - Agent selection  |       |  + OCR               |       |                     |
|  - Chat interface   |       +----------------------+       |  - pyautogui        |
|  - LLM API calls    |                                      |  - Screenshot       |
|  - Tool execution   |<------------------------------------>|  - Action execution |
+---------------------+                                      +---------------------+
```

1. **OmniParser Server** (GPU) - Parses screenshots into structured UI elements
2. **OmniBox** (CPU) - Windows 11 VM running in Docker with a Flask control server
3. **Gradio UI + Agent Loop** (CPU) - User interface and LLM-driven agent orchestration

---

## Directory Structure

```
vision_omni/
├── omnitool/                          # Main application
│   ├── gradio/                        # UI and agent orchestration
│   │   ├── app.py                     # Primary Gradio UI entry point (port 7888)
│   │   ├── app_new.py                 # Alternative UI implementation
│   │   ├── app_streamlit.py           # Streamlit-based alternative UI
│   │   ├── loop.py                    # Agentic sampling loop (LLM orchestration)
│   │   ├── agent/                     # Agent implementations
│   │   │   ├── anthropic_agent.py     # Claude agent (native computer use)
│   │   │   ├── vlm_agent.py           # Vision-LLM agent (GPT-4o, R1, Qwen)
│   │   │   ├── vlm_agent_with_orchestrator.py  # Multi-step planning agent
│   │   │   └── llm_utils/            # LLM client wrappers
│   │   │       ├── oaiclient.py       # OpenAI API client
│   │   │       ├── groqclient.py      # Groq/DeepSeek R1 client
│   │   │       ├── omniparserclient.py# OmniParser server client
│   │   │       └── utils.py           # Shared utilities
│   │   ├── executor/
│   │   │   └── anthropic_executor.py  # Processes tool calls and collects results
│   │   ├── tools/                     # Tool implementations
│   │   │   ├── base.py               # ToolResult, ToolError base classes
│   │   │   ├── computer.py           # Computer control (mouse, keyboard, screenshot)
│   │   │   ├── screen_capture.py     # Screenshot capture from Flask server
│   │   │   ├── collection.py         # Tool collection management
│   │   │   └── config.py             # Flask server URL config
│   │   └── tmp/                       # Temporary files (screenshots, SOM images)
│   ├── omniparserserver/
│   │   └── omniparserserver.py        # FastAPI server for screen parsing
│   └── omnibox/                       # Windows VM infrastructure
│       ├── Dockerfile                 # Docker image (QEMU-based)
│       ├── compose.yml                # Docker Compose config
│       ├── scripts/
│       │   └── manage_vm.sh           # VM lifecycle management
│       └── vm/
│           ├── win11iso/              # Windows 11 ISO storage
│           ├── win11storage/          # VM disk image
│           └── win11setup/
│               └── setupscripts/
│                   ├── setup.ps1      # PowerShell VM setup
│                   └── server/
│                       └── main.py    # Flask server inside the VM
├── util/                              # Core parsing utilities
│   ├── omniparser.py                  # OmniParser class (orchestrates parsing)
│   ├── utils.py                       # Model loading, OCR, annotation (~2800 lines)
│   └── box_annotator.py              # Bounding box visualization
├── eval/                              # Benchmarking
│   ├── ss_pro_gpt4o_omniv2.py        # Screen Spot Pro evaluation
│   └── logs_sspro_omniv2.json        # Evaluation results
├── weights/                           # ML model weights
│   ├── icon_detect/                   # YOLO icon detection model
│   └── icon_caption_florence/         # Florence-2 caption model
├── docs/                              # Documentation
├── imgs/                              # Image assets
├── gradio_demo.py                     # Standalone OmniParser demo UI
├── demo.ipynb                         # Jupyter notebook examples
├── main.py                            # Simple entry point
├── pyproject.toml                     # Project metadata & dependencies
├── requirements.txt                   # Python dependencies
└── README.md                          # Project README
```

---

## Core Components

### 1. OmniParser Server

**File:** `omnitool/omniparserserver/omniparserserver.py`

A FastAPI server that receives screenshots and returns structured UI element data.

**Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/parse/` | POST | Accepts base64 image, returns SOM-annotated image + parsed element list |
| `/probe/` | GET | Health check |

**Parsing Pipeline** (handled by `util/omniparser.py` and `util/utils.py`):

1. **OCR Detection** (`check_ocr_box`) - Detects text using EasyOCR or PaddleOCR
2. **Icon Detection** (`get_yolo_model`) - Detects UI icons/buttons using YOLO v8
3. **Icon Captioning** (`get_parsed_content_icon`) - Generates descriptions for each icon using Florence-2
4. **Overlap Removal** - Removes duplicate detections between OCR and icon detection
5. **Image Annotation** (`get_som_labeled_img`) - Overlays numbered labels on the screenshot (SOM image)
6. **Content Generation** - Returns structured list of parsed elements

**Parsed Element Format:**
```json
{
  "type": "text | icon",
  "content": "descriptive text or icon caption",
  "bbox": [x_min, y_min, x_max, y_max],
  "idx": 0
}
```

**Startup:**
```bash
python -m omniparserserver \
  --som_model_path ../../weights/icon_detect/model.pt \
  --caption_model_name florence2 \
  --caption_model_path ../../weights/icon_caption_florence \
  --device cuda \
  --BOX_TRESHOLD 0.05 \
  --port 8000
```

---

### 2. OmniBox (Windows 11 VM)

**Directory:** `omnitool/omnibox/`

A Docker container running a lightweight Windows 11 Enterprise VM (~5GB) with:
- **NoVNC viewer** on port 8006 for remote desktop access
- **Flask server** on port 5006 for programmatic computer control

**Flask Server Endpoints** (`vm/win11setup/setupscripts/server/main.py`):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/probe` | GET | Health check |
| `/screenshot` | GET | Capture screen with cursor overlay |
| `/screen_size` | GET | Return display dimensions |
| `/mouse_position` | GET | Current cursor coordinates |
| `/action` | POST | Execute pyautogui actions |
| `/execute` | POST | Run arbitrary commands |

**Supported pyautogui Actions:**
`moveTo`, `click`, `rightClick`, `middleClick`, `doubleClick`, `dragTo`, `keyDown`, `keyUp`, `typewrite`, `press`, `scroll`, `mouseDown`, `mouseUp`

**Docker Compose Resources:**
- RAM: 8GB
- CPU: 4 cores
- Disk: 20GB
- Ports: 8006 (VNC), 5006 (Flask API)

---

### 3. Gradio UI

**File:** `omnitool/gradio/app.py`

The primary user interface running on port 7888. Provides:
- Model and provider selection (dropdown)
- API key input
- Chat interface with Send/Stop controls
- Optional NoVNC iframe for viewing the Windows desktop
- Configurable "N most recent images" slider for context window management

**State Management:**
```python
{
    "messages": [],              # Conversation history
    "model": "omniparser + gpt-4o",
    "provider": "openai",
    "api_key": "",
    "chatbot_messages": [],      # UI display messages
    "only_n_most_recent_images": 2,
    "stop": False
}
```

**Startup:**
```bash
python app.py \
  --windows_host_url localhost:8006 \
  --omniparser_server_url localhost:8000 \
  --flask_server_url localhost:5006
```

---

### 4. Agent Sampling Loop

**File:** `omnitool/gradio/loop.py`

The `sampling_loop_sync()` function is the core orchestration loop. It:

1. Receives user input from Gradio
2. Routes to the appropriate agent based on model selection
3. Sends screenshots + parsed screen info to the LLM
4. Receives action responses (mouse/keyboard commands)
5. Executes actions via the AnthropicExecutor
6. Captures new screenshots and re-parses them
7. Loops until the task is complete or the user stops

**Agent Routing:**
- Claude models -> `AnthropicActor` + `AnthropicExecutor`
- VLM models (GPT-4o, o1, o3-mini, R1, Qwen) -> `VLMAgent`
- VLM models with orchestration suffix -> `VLMOrchestratedAgent`

---

### 5. Agents

#### AnthropicActor (`agent/anthropic_agent.py`)
- Uses Anthropic's native **Computer Use** beta API (`computer-use-2024-10-22`)
- Supports three providers: Anthropic Direct, AWS Bedrock, Google Vertex AI
- Claude generates tool calls (mouse, keyboard, screenshot) natively
- Token usage and cost tracking built-in

#### VLMAgent (`agent/vlm_agent.py`)
- Works with vision-capable LLMs that don't have native computer-use support
- Injects parsed screen info (from OmniParser) into the system prompt
- LLM outputs a JSON action:
  ```json
  {
    "Reasoning": "analysis of the current screen state",
    "Next Action": "left_click | type | scroll_up | scroll_down | wait | None",
    "Box ID": 5,
    "value": "text to type (if action is type)"
  }
  ```
- Converts JSON responses to tool-use blocks for the AnthropicExecutor
- Supports: OpenAI (GPT-4o, o1, o3-mini), Groq (DeepSeek R1), DashScope (Qwen 2.5VL)

#### VLMOrchestratedAgent (`agent/vlm_agent_with_orchestrator.py`)
- Extends VLMAgent with multi-step task planning and progress tracking
- **Plan Phase:** Decomposes the task into numbered subgoals via LLM
- **Execution Phase:** Executes one action per step (same as VLMAgent)
- **Ledger Phase:** After each step, an orchestrator LLM evaluates:
  - Is the request satisfied?
  - Are we stuck in a loop?
  - Is forward progress being made?
  - What refined instruction should come next?
- Saves full trajectory (screenshots, SOM images, LLM responses, latencies) to disk

---

### 6. Tools

#### ComputerTool (`tools/computer.py`)
The primary tool for interacting with the Windows VM.

**Actions:**
`key`, `type`, `mouse_move`, `left_click`, `left_click_drag`, `right_click`, `middle_click`, `double_click`, `screenshot`, `cursor_position`, `hover`, `wait`, `scroll_up`, `scroll_down`

**Communication:** HTTP POST to the Flask server's `/action` endpoint with:
```json
{
  "action": "click",
  "args": [500, 300],
  "kwargs": {}
}
```

**Coordinate Scaling:** Supports resolution mapping between API coordinates and actual screen:
- XGA: 1024x768
- WXGA: 1280x800
- FWXGA: 1366x768

#### ScreenCapture (`tools/screen_capture.py`)
- Captures screenshots via HTTP GET to the Flask server
- Saves to `./tmp/outputs/screenshot_{uuid}.png`

#### ToolResult (`tools/base.py`)
- Dataclass with fields: `output`, `error`, `base64_image`, `system`
- Supports combining results via the `+` operator

---

### 7. LLM Clients

#### OpenAI Client (`agent/llm_utils/oaiclient.py`)
- Wraps OpenAI API for GPT-4o, o1, o3-mini
- Converts Anthropic message format to OpenAI format
- Handles image embedding as base64 data URLs
- Special handling for reasoning models (o1, o3-mini) with `reasoning_effort` parameter

#### Groq Client (`agent/llm_utils/groqclient.py`)
- Wraps Groq API for DeepSeek R1
- Strips images from messages (R1 is text-only)
- Parses `<think>` and `<output>` reasoning tags

#### OmniParser Client (`agent/llm_utils/omniparserclient.py`)
- Captures screenshot from the Flask server
- Sends to OmniParser server `/parse/` endpoint
- Returns SOM image, parsed content list, and screen_info string:
  ```
  ID: 0, Text: "Save Button"
  ID: 1, Icon: "checkbox"
  ID: 2, Text: "Search bar"
  ```

---

### 8. Executor

**File:** `executor/anthropic_executor.py`

The `AnthropicExecutor` processes LLM responses:
1. Iterates over content blocks in the LLM response
2. For `text` blocks: yields them to the UI
3. For `tool_use` blocks: runs the corresponding tool via `ToolCollection`
4. Catches `ToolError` exceptions and converts them to `ToolFailure` results
5. Constructs `BetaToolResultBlockParam` with output/error/image for the next loop iteration

---

## Data Flow

```
User types task in Gradio UI (port 7888)
         |
         v
sampling_loop_sync() starts
         |
         v
OmniParserClient captures screenshot from Flask (port 5006)
         |
         v
Screenshot sent to OmniParser Server (port 8000)
    |-- YOLO detects icons/buttons
    |-- OCR detects text elements
    |-- Florence-2 captions each icon
    |-- Returns: SOM image + parsed_content_list
         |
         v
Agent receives parsed screen + conversation history
    |-- Builds system prompt with UI element descriptions
    |-- Calls LLM API (Anthropic/OpenAI/Groq/DashScope)
    |-- LLM returns next action (mouse/keyboard/screenshot)
         |
         v
AnthropicExecutor processes action
    |-- ComputerTool sends HTTP POST to Flask /action
    |-- Flask server executes pyautogui command in Windows VM
    |-- New screenshot captured
         |
         v
Loop repeats until task complete or user stops
```

---

## Supported Models and Providers

| Provider | Models | Agent Type |
|----------|--------|------------|
| Anthropic (Direct) | Claude 3.5 Sonnet | AnthropicActor |
| Anthropic (Bedrock) | Claude 3.5 Sonnet | AnthropicActor |
| Anthropic (Vertex) | Claude 3.5 Sonnet | AnthropicActor |
| OpenAI | GPT-4o, o1, o3-mini | VLMAgent |
| Groq | DeepSeek R1 | VLMAgent |
| DashScope | Qwen 2.5VL | VLMAgent |

All models can optionally use the `VLMOrchestratedAgent` for multi-step planning.

---

## Configuration and Startup

### Full System Startup (3 components)

**1. OmniParser Server (GPU machine):**
```bash
cd omnitool/omniparserserver
python -m omniparserserver \
  --som_model_path ../../weights/icon_detect/model.pt \
  --caption_model_name florence2 \
  --caption_model_path ../../weights/icon_caption_florence \
  --device cuda \
  --BOX_TRESHOLD 0.05 \
  --port 8000
```

**2. OmniBox VM (Docker host):**
```bash
cd omnitool/omnibox
docker compose up -d
```

**3. Gradio UI:**
```bash
cd omnitool/gradio
python app.py \
  --windows_host_url <vm-host>:8006 \
  --omniparser_server_url <gpu-host>:8000 \
  --flask_server_url <vm-host>:5006
```

### Standalone Demo (no VM needed)
```bash
python gradio_demo.py
```
Upload screenshots directly to test OmniParser parsing without the full agent loop.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `omnitool/omniparserserver/omniparserserver.py` | FastAPI server for screen parsing |
| `util/omniparser.py` | OmniParser class orchestrating YOLO + OCR + captioning |
| `util/utils.py` | Model loading, OCR, annotation, overlap removal |
| `omnitool/gradio/app.py` | Main Gradio UI and state management |
| `omnitool/gradio/loop.py` | Sampling loop routing to agents |
| `omnitool/gradio/agent/anthropic_agent.py` | Claude computer-use agent |
| `omnitool/gradio/agent/vlm_agent.py` | Vision-LLM agent (OpenAI, Groq, Qwen) |
| `omnitool/gradio/agent/vlm_agent_with_orchestrator.py` | Orchestrated multi-step agent |
| `omnitool/gradio/executor/anthropic_executor.py` | Tool execution and result collection |
| `omnitool/gradio/tools/computer.py` | Mouse, keyboard, screenshot actions |
| `omnitool/gradio/tools/screen_capture.py` | Screenshot capture from Flask server |
| `omnitool/gradio/agent/llm_utils/oaiclient.py` | OpenAI API wrapper |
| `omnitool/gradio/agent/llm_utils/groqclient.py` | Groq/DeepSeek R1 wrapper |
| `omnitool/gradio/agent/llm_utils/omniparserclient.py` | OmniParser server client |
| `omnitool/omnibox/compose.yml` | Docker Compose for Windows VM |
| `omnitool/omnibox/vm/win11setup/setupscripts/server/main.py` | Flask server running inside VM |
| `omnitool/omnibox/scripts/manage_vm.sh` | VM lifecycle management script |
| `gradio_demo.py` | Standalone OmniParser demo (no VM) |
| `weights/icon_detect/model.pt` | YOLO icon detection model |
| `weights/icon_caption_florence/` | Florence-2 caption model |

---

## Dependencies

**Core ML:** PyTorch, Ultralytics (YOLO v8), Transformers (Florence-2), EasyOCR, PaddleOCR

**Web Frameworks:** FastAPI + Uvicorn, Gradio, Flask (in VM)

**LLM SDKs:** anthropic, openai, groq, boto3 (Bedrock), google-auth (Vertex), dashscope (Qwen)

**Automation:** pyautogui, uiautomation, screeninfo

**Python Version:** 3.12
