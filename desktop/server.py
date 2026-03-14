"""
Desktop control server - runs on the PC being controlled.
Provides endpoints for mouse/keyboard control and screenshot capture.

Usage:
    uv run server.py --port 5010
    uv run server.py --port 5010 --ngrok
"""

import os
import sys
import logging
import argparse
import threading
import traceback
from io import BytesIO

from flask import Flask, request, jsonify, send_file
import pyautogui
from PIL import Image

pyautogui.FAILSAFE = False

parser = argparse.ArgumentParser(description="Desktop control server")
parser.add_argument("--port", type=int, default=5010, help="Port to listen on")
parser.add_argument("--ngrok", action="store_true", help="Start ngrok tunnel for remote access")
parser.add_argument("--ngrok-auth-token", type=str, default=None, help="Ngrok auth token (or set NGROK_AUTHTOKEN env var)")
parser.add_argument("--log-file", type=str, default=None, help="Log file path (default: stdout)")
args = parser.parse_args()

if args.log_file:
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG, filemode='w')
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
control_lock = threading.Lock()

# Load cursor image for screenshot overlay (optional)
cursor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cursor.png")
cursor_image = None
if os.path.exists(cursor_path):
    try:
        cursor_image = Image.open(cursor_path)
        cursor_image = cursor_image.resize((int(cursor_image.width / 1.5), int(cursor_image.height / 1.5)))
    except Exception as e:
        logger.warning(f"Could not load cursor image: {e}")


@app.route('/probe', methods=['GET'])
def probe():
    return jsonify({"status": "ok", "message": "Desktop server is running"}), 200


@app.route('/screenshot', methods=['GET'])
def screenshot():
    with control_lock:
        try:
            img = pyautogui.screenshot()
            if cursor_image:
                x, y = pyautogui.position()
                img.paste(cursor_image, (x, y), cursor_image)
            buf = BytesIO()
            img.save(buf, 'PNG')
            buf.seek(0)
            return send_file(buf, mimetype='image/png')
        except Exception as e:
            logger.error(traceback.format_exc())
            return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/screen_size', methods=['GET'])
def screen_size():
    with control_lock:
        size = pyautogui.size()
        return jsonify({"width": size.width, "height": size.height})


@app.route('/mouse_position', methods=['GET'])
def mouse_position():
    with control_lock:
        pos = pyautogui.position()
        return jsonify({"x": pos.x, "y": pos.y})


@app.route('/action', methods=['POST'])
def action():
    """Execute a pyautogui action.

    JSON body: {"action": "click"|"moveTo"|..., "args": [...], "kwargs": {...}}
    """
    with control_lock:
        try:
            data = request.json
            action_name = data.get('action')
            action_args = data.get('args', [])
            action_kwargs = data.get('kwargs', {})

            func = getattr(pyautogui, action_name, None)
            if func is None:
                return jsonify({"status": "error", "message": f"Unknown action: {action_name}"}), 400

            result = func(*action_args, **action_kwargs)
            return jsonify({"status": "success", "result": str(result) if result is not None else None})
        except Exception as e:
            logger.error(traceback.format_exc())
            return jsonify({"status": "error", "message": str(e)}), 500


def self_test():
    """Verify the desktop environment is accessible."""
    print("Running self-test...")
    try:
        size = pyautogui.size()
        assert size.width > 0 and size.height > 0, "Invalid screen size"
        print(f"  Screen size: {size.width}x{size.height}")
    except Exception as e:
        print(f"  FAILED: Could not get screen size: {e}")
        print("  Make sure you're running on a machine with a display.")
        sys.exit(1)

    try:
        pos = pyautogui.position()
        print(f"  Mouse position: ({pos.x}, {pos.y})")
    except Exception as e:
        print(f"  FAILED: Could not get mouse position: {e}")
        sys.exit(1)

    try:
        img = pyautogui.screenshot()
        assert img.size[0] > 0, "Invalid screenshot"
        print(f"  Screenshot capture: OK ({img.size[0]}x{img.size[1]})")
    except Exception as e:
        print(f"  FAILED: Could not capture screenshot: {e}")
        sys.exit(1)

    print("Self-test passed!")


if __name__ == '__main__':
    self_test()

    if args.ngrok:
        from pyngrok import ngrok
        token = args.ngrok_auth_token or os.environ.get("NGROK_AUTHTOKEN")
        if token:
            ngrok.set_auth_token(token)
        tunnel = ngrok.connect(args.port, "http")
        print(f"Ngrok tunnel: {tunnel.public_url}")

    print(f"Starting desktop server on 0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port)
