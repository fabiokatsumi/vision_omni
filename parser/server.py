"""
OmniParser server - processes screenshots with YOLO + OCR + Florence-2.
Runs on a PC with CUDA GPU.

Usage:
    uv run server.py --port 8013 --device cuda
    uv run server.py --port 8013 --device cuda --ngrok
"""

import os
import sys
import time
import base64
import argparse

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


def parse_arguments():
    parser = argparse.ArgumentParser(description='OmniParser server')
    parser.add_argument('--som-model-path', type=str, default=None, help='Path to YOLO model (default: ../weights/icon_detect/model.pt)')
    parser.add_argument('--caption-model-name', type=str, default='florence2', help='Caption model name')
    parser.add_argument('--caption-model-path', type=str, default=None, help='Path to caption model (default: ../weights/icon_caption_florence)')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cuda or cpu')
    parser.add_argument('--box-threshold', type=float, default=0.05, help='Detection threshold')
    parser.add_argument('--weights-dir', type=str, default='../weights', help='Weights directory')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--ngrok', action='store_true', help='Start ngrok tunnel')
    parser.add_argument('--ngrok-auth-token', type=str, default=None, help='Ngrok auth token')
    return parser.parse_args()


args = parse_arguments()

# Resolve default paths relative to weights dir
weights_dir = os.path.abspath(args.weights_dir)
som_model_path = args.som_model_path or os.path.join(weights_dir, 'icon_detect', 'model.pt')
caption_model_path = args.caption_model_path or os.path.join(weights_dir, 'icon_caption_florence')

# Auto-download weights if missing
from setup_weights import ensure_weights
ensure_weights(weights_dir)

# Verify weights exist
if not os.path.exists(som_model_path):
    print(f"ERROR: YOLO model not found at {som_model_path}")
    sys.exit(1)
if not os.path.exists(caption_model_path):
    print(f"ERROR: Caption model not found at {caption_model_path}")
    sys.exit(1)

# Initialize parser
config = {
    'som_model_path': som_model_path,
    'caption_model_name': args.caption_model_name,
    'caption_model_path': caption_model_path,
    'BOX_TRESHOLD': args.box_threshold,
}

print("Loading models...")
from omniparser import Omniparser
omniparser = Omniparser(config)

# Self-test: parse a synthetic image
print("Running self-test...")
try:
    from PIL import Image
    import io
    test_img = Image.new('RGB', (200, 200), color='white')
    buf = io.BytesIO()
    test_img.save(buf, format='PNG')
    test_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    result = omniparser.parse(test_b64)
    assert result is not None, "Expected a result tuple"
    som_img, parsed_list = result
    assert isinstance(som_img, str), "Expected base64 string"
    assert isinstance(parsed_list, list), "Expected list"
    print(f"Self-test passed! (detected {len(parsed_list)} elements)")
except Exception as e:
    print(f"Self-test FAILED: {e}")
    sys.exit(1)

# Create FastAPI app
app = FastAPI()


class ParseRequest(BaseModel):
    base64_image: str


@app.post("/parse/")
async def parse(req: ParseRequest):
    print('start parsing...')
    start = time.time()
    som_img, parsed_content_list = omniparser.parse(req.base64_image)
    latency = time.time() - start
    print(f'parse time: {latency:.2f}s')
    return {
        "som_image_base64": som_img,
        "parsed_content_list": parsed_content_list,
        "latency": latency,
    }


@app.get("/probe/")
async def probe():
    return {"status": "ok", "message": "Parser server is running"}


if __name__ == "__main__":
    if args.ngrok:
        from pyngrok import ngrok
        token = args.ngrok_auth_token or os.environ.get("NGROK_AUTHTOKEN")
        if token:
            ngrok.set_auth_token(token)
        tunnel = ngrok.connect(args.port, "http")
        print(f"Ngrok tunnel: {tunnel.public_url}")

    print(f"Starting parser server on {args.host}:{args.port}")
    uvicorn.run("server:app", host=args.host, port=args.port)
