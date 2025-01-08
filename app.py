from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import json
import subprocess

app = Flask(__name__)

# Directory for saving files
UPLOAD_FOLDER = "./"
CONFIG_FILE = "configurations.json"
OUTLINE_FILE = "outline.json"

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tracking processing state
processing_state = {"is_processing": False}

def delete_files():
    # List of files to delete
    files_to_delete = [
        "scene_1.wav",
        "scene_1_output.mp4",
        "scene_2.wav",
        "scene_2_output.mp4",
        "scene_3.wav",
        "scene_3_output.mp4",
        "scene_4.wav",
        "scene_4_output.mp4",
        "final_output.mp4"
    ]

    for filename in files_to_delete:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/upload_config', methods=['POST'])
def upload_config():
    global processing_state

    # Accept configuration parameters from JSON payload
    try:
        config_data = request.get_json()
        if not config_data:
            return jsonify({"error": "No JSON payload provided."}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400

    # Save the configuration to a file
    config_path = os.path.join(UPLOAD_FOLDER, CONFIG_FILE)
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=4)

    # Delete specified files
    delete_files()

    # Mark as processing
    processing_state["is_processing"] = True

    # Execute a separate Python file for processing
    try:
        subprocess.run(["python", "voice.py"], check=True)
        processing_state["is_processing"] = False
    except subprocess.CalledProcessError as e:
        processing_state["is_processing"] = False
        return jsonify({"error": "Processing failed.", "details": str(e)}), 500

    return jsonify({"message": "Configuration received and processing started.", "config": config_data}), 200

@app.route('/processing_status', methods=['GET'])
def processing_status():
    # Check and return the processing status
    if processing_state["is_processing"]:
        return jsonify({"status": True}), 200  # Still processing
    else:
        return jsonify({"status": False}), 200  # Processing complete

@app.route('/read_outline', methods=['GET'])
def read_outline():
    outline_path = os.path.join(UPLOAD_FOLDER, OUTLINE_FILE)
    if os.path.exists(outline_path):
        try:
            with open(outline_path, 'r') as f:
                outline_data = json.load(f)
            return jsonify(outline_data), 200
        except Exception as e:
            return jsonify({"error": "Failed to read outline file.", "details": str(e)}), 500
    else:
        return jsonify({"error": "Outline file not found."}), 404

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    # Accept a PDF file
    pdf_file = request.files.get('pdf')
    if pdf_file:
        # Save the file
        pdf_filename = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        pdf_file.save(pdf_path)

        return jsonify({"message": "PDF uploaded successfully."}), 200
    else:
        return jsonify({"error": "No PDF file provided."}), 400

# Simulate processing completion (for testing purposes)
@app.route('/complete_processing', methods=['POST'])
def complete_processing():
    global processing_state
    processing_state["is_processing"] = False
    return jsonify({"message": "Processing marked as complete."}), 200

if __name__ == '__main__':
    app.run(debug=True)

