from flask import Flask, request, jsonify
import sys

# Initialize the Flask application
app = Flask(__name__)

# This route listens for POST requests at /process
@app.route('/process', methods=['POST'])
def process_data():
    try:
        # 1. Get the JSON sent from Mac
        data = request.json
        print(f"Received request: {data}", flush=True) # Visible in VS Code

        # --- YOUR COMPLEX LOGIC GOES HERE ---
        # (You can print as much as you want here for debugging)
        print("Starting complex calculation...", flush=True)
        
        # Example logic
        epochs = data.get('params', {}).get('epochs', 1)
        result_val = epochs * 100 
        
        print(f"Calculation finished. Result: {result_val}", flush=True)
        # ------------------------------------

        response = {
            "status": "success",
            "result": result_val
        }
        return jsonify(response)

    except Exception as e:
        print(f" Error occurred: {e}", file=sys.stderr)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Run the server on localhost port 5000
    print(" Server is running on Ubuntu. Waiting for requests...", flush=True)
    app.run(host='127.0.0.1', port=5000)