import os
import logging
import argparse
import shlex
import subprocess
from flask import Flask, request, jsonify, send_file
import threading
import traceback
import pyautogui
from PIL import Image
from io import BytesIO

pyautogui.FAILSAFE = False


def execute_anything(data):
    """Execute any command received in the JSON request.
    WARNING: This function executes commands without any safety checks."""
    # The 'command' key in the JSON request should contain the command to be executed.
    shell = data.get('shell', False)
    command = data.get('command', "" if shell else [])

    if isinstance(command, str) and not shell:
        command = shlex.split(command)

    # Expand user directory
    for i, arg in enumerate(command):
        if arg.startswith("~/"):
            command[i] = os.path.expanduser(arg)

    # Execute the command without any safety checks.
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell, text=True, timeout=120)
        return jsonify({
            'status': 'success',
            'output': result.stdout,
            'error': result.stderr,
            'returncode': result.returncode
        })
    except Exception as e:
        logger.error("\n" + traceback.format_exc() + "\n")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


def execute(data):
    """Action space aware implementation. Should not use arbitrary code execution."""
    return jsonify({
        'status': 'error',
        'message': 'Not implemented. Please add your implementation to omnitool/omnibox/vm/win11setup/setupscripts/server/main.py.'
    }), 500


execute_impl = execute_anything   # switch to execute_anything to allow any command. Please use with caution only for testing purposes.


parser = argparse.ArgumentParser()
parser.add_argument("--log_file", help="log file path", type=str,
                    default=os.path.join(os.path.dirname(__file__), "server.log"))
parser.add_argument("--port", help="port", type=int, default=5006)
args = parser.parse_args()

logging.basicConfig(filename=args.log_file,level=logging.DEBUG, filemode='w' )
logger = logging.getLogger('werkzeug')

app = Flask(__name__)

computer_control_lock = threading.Lock()

@app.route('/probe', methods=['GET'])
def probe_endpoint():
    return jsonify({"status": "Probe successful", "message": "Service is operational"}), 200

@app.route('/execute', methods=['POST'])
def execute_command():
    # Only execute one command at a time
    with computer_control_lock:
        data = request.json
        return execute_impl(data)

@app.route('/screenshot', methods=['GET'])
def capture_screen_with_cursor():
    cursor_path = os.path.join(os.path.dirname(__file__), "cursor.png")
    screenshot = pyautogui.screenshot()
    cursor_x, cursor_y = pyautogui.position()
    cursor = Image.open(cursor_path)
    # make the cursor smaller
    cursor = cursor.resize((int(cursor.width / 1.5), int(cursor.height / 1.5)))
    screenshot.paste(cursor, (cursor_x, cursor_y), cursor)

    # Convert PIL Image to bytes and send
    img_io = BytesIO()
    screenshot.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

@app.route('/screen_size', methods=['GET'])
def screen_size():
    """Return the screen dimensions."""
    with computer_control_lock:
        size = pyautogui.size()
        return jsonify({"width": size.width, "height": size.height})

@app.route('/mouse_position', methods=['GET'])
def mouse_position():
    """Return the current mouse cursor position."""
    with computer_control_lock:
        pos = pyautogui.position()
        return jsonify({"x": pos.x, "y": pos.y})

@app.route('/action', methods=['POST'])
def perform_action():
    """Execute a pyautogui action directly.

    Expected JSON body:
    {
        "action": "moveTo" | "click" | "rightClick" | "middleClick" | "doubleClick" |
                  "dragTo" | "keyDown" | "keyUp" | "typewrite" | "press" | "scroll" |
                  "mouseDown" | "mouseUp",
        "args": [...],      # positional arguments
        "kwargs": {...}     # keyword arguments
    }
    """
    with computer_control_lock:
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
            logger.error("\n" + traceback.format_exc() + "\n")
            return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=args.port)
