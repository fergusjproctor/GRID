"""
Script that loads in a sg, preprocesses the data and runs GRID inference on it, refactored from original plan to make as lightweight as possible before tryng to integrate with VH
"""

import sys
import json

from flask import Flask, request, jsonify

from plan import run_inference, load_inference


# Initialize the Flask application
app = Flask(__name__)


# This route listens for POST requests at /process
@app.route('/process', methods=['POST'])
def process_data():
    try:

        data = request.json
        print(f"Received request: {data.get('params', {}).get('instr', 1)}") # Visible in VS Code

        sg_json = data.get('params', {}).get('scene_graph', {})
        rg_json = data.get('params', {}).get('robot_graph', {})
        instr = data.get('params', {}).get('instr', 1)
        raw_input = {"instr": instr, "rg_json": rg_json, "sg_json": sg_json}

        # save the raw input for debugging
        with open("raw_input_debug.json", 'w') as f:
            json.dump(raw_input, f, indent=4)


        action, object, object_id = run_inference(instructor_model, GRID_model, raw_input)
        print(f"{action} {object} {object_id}")
        response = {
            "status": "success",
            "result": {
                "action": f"{action}",
                "object_label": f"{object}",
              "grid_object_id": f"{object_id}"
              }
        }
        return jsonify(response)

    except Exception as e:
        print(f" Error occurred: {e}", file=sys.stderr)
        return jsonify({"status": "error", "message": str(e)}), 500
        


if __name__ == "__main__":
    instructor_model, GRID_model = load_inference()
    print(" Server is running on Ubuntu. Waiting for requests...", flush=True)
    app.run(host='127.0.0.1', port=5000)