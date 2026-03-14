# OmniTool

AI agent that controls any desktop via screen parsing and vision-language models.

Based on [OmniParser](https://arxiv.org/abs/2408.00203) by Microsoft Research.

## Architecture

```
┌─────────┐     screenshots    ┌─────────┐     parsed screen    ┌─────────┐
│ Desktop │ ──────────────────▶│ Parser  │ ────────────────────▶│  Agent  │
│ server  │ ◀──────────────── │ server  │                      │  (UI)   │
│         │   mouse/keyboard   │ (GPU)   │                      │         │
└─────────┘                    └─────────┘                      └─────────┘
 Any PC(s)                    GPU machine                    Orchestrator PC
```

- **`desktop/`** - Runs on each PC you want to control. Captures screenshots, executes mouse/keyboard actions.
- **`parser/`** - Runs on a GPU machine. Processes screenshots with YOLO + OCR + Florence-2.
- **`agent/`** - Runs anywhere. Gradio UI + AI agent loop. Connects to desktop + parser servers.

One parser + one agent can control **many desktops**.

## Quick Start

### 1. Desktop server (on each PC to control)

```bash
cd desktop
uv run server.py --port 5010
```

### 2. Parser server (on GPU machine)

```bash
cd parser
uv run server.py --port 8013 --device cuda
```

Weights are **auto-downloaded** from HuggingFace on first run.

### 3. Agent (on orchestrator PC)

```bash
cd agent
uv run app.py --desktops 192.168.1.10:5010 --parser-url 10.0.0.5:8013
```

Opens Gradio UI at `http://localhost:7888`.

## Remote Access with Ngrok

Each server supports `--ngrok` to create a public tunnel:

```bash
# On desktop PC
cd desktop && uv run server.py --port 5010 --ngrok

# On GPU machine
cd parser && uv run server.py --port 8013 --device cuda --ngrok

# On agent PC (use the ngrok URLs printed by the other servers)
cd agent && uv run app.py --desktops https://abc123.ngrok-free.app --parser-url https://def456.ngrok-free.app --ngrok
```

Set `NGROK_AUTHTOKEN` env var or pass `--ngrok-auth-token`.

## Multi-PC Setup

Control multiple desktops by passing comma-separated URLs:

```bash
cd agent
uv run app.py --desktops pc1:5010,pc2:5010,pc3:5010 --parser-url gpu:8013
```

A dropdown in the UI lets you switch between desktops.

## Supported Models

| Provider | Models | Type |
|----------|--------|------|
| OpenAI | GPT-4o, o1, o3-mini | VLM Agent |
| Anthropic | Claude 3.5 Sonnet | Native Computer Use |
| Groq | DeepSeek R1 | VLM Agent |
| DashScope | Qwen 2.5VL | VLM Agent |

All models also support an `-orchestrated` variant with multi-step planning.

## License

MIT License. See [LICENSE](LICENSE).

Icon detection model (YOLO) is under AGPL license. Caption model (Florence-2) is under MIT license.

## Citation

```bibtex
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent},
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
