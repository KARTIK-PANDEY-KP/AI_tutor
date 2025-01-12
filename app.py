from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import json
import subprocess
import glob
from helper import semantic_parts, process_pdf
from openai import OpenAI
from helper import upload_to_the_vector_database
from time import sleep
from pinecone import Pinecone



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
    import glob

    # Define file patterns to delete
    file_patterns = [
        "scene_*.wav",
        "scene_*.mp4",
        "final_output.mp4"
    ]

    # Iterate over file patterns and delete matching files
    for pattern in file_patterns:
        for file_path in glob.glob(pattern):  # Match files using the pattern
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except FileNotFoundError:
                print(f"File not found (skipping): {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")


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
#     delete_files()

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

@app.route('/semantic_parts', methods=['POST'])
def semantic_parts_endpoint():
    """
    Endpoint to process only 'user_query' using the `semantic_parts` function.
    """
    try:
        # Get the JSON payload from the user
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON payload provided."}), 400

        # Extract 'user_query'
        user_query = data.get("user_query")
        if not user_query:
            return jsonify({"error": "'user_query' is required."}), 400

        # Process the 'user_query' (example usage with the semantic_parts function)
        # For this simplified example, let's assume other inputs are fixed.
        response = semantic_parts(
            client=OpenAI(),  # Replace with your client object
            prompt_textual=user_query,  # Using 'user_query' directly as the prompt
        )

        # Return the generated response
        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

#new
@app.route('/upload_pdf_get_sum', methods=['POST'])
def upload_pdf_get_sum():
    # Accept a PDF file
    pdf_file = request.files.get('pdf')
    if pdf_file:
        # Save the file
        pdf_filename = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        pdf_file.save(pdf_path)

        # Set the embedding model and its parameters
        model_name = "text-embedding-3-small"
        max_tokens = 8191
        dimensions = 1536

        # I think it uploads to the vectorDB, NOT SURE IF IT APPEND THE DATA OR OVERWRITES THE EXISTING DATA
        ret = upload_to_the_vector_database(pdf_path, model_name, max_tokens, dimensions)
        print("------ UPLOADED -----------")
        sleep(20)
        process_pdf
        print("------ DELETED -----------")

        # Set up Pinecone client
        pinecone_client = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY")
        )
    #     pinecone_client.delete_index("test")

        index = pinecone_client.Index(host = "https://test-gfkht3t.svc.aped-4627-b74a.pinecone.io")
    #     index.delete(delete_all=True)
        index.delete(delete_all=True)
        
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)  # Delete the file
                print(f"{pdf_path} has been deleted successfully.")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
             print(f"{pdf_path} does not exist.")
    
        return jsonify({"message": "PDF uploaded successfully."}), 200
    
    else:
        
        index.delete(delete_all=True)
            
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)  # Delete the file
                print(f"{pdf_path} has been deleted successfully.")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
             print(f"{pdf_path} does not exist.")

        return jsonify({"message": "PDF uploaded successfully."}), 200
            
        return jsonify({"error": "No PDF file provided."}), 400
    
@app.route('/upload_pdf_get_sum_graph', methods=['POST'])
def upload_pdf_get_sum_graph():
    # Accept a PDF file
    pdf_file = request.files.get('pdf')
    if pdf_file:
        # Save the file
        pdf_filename = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        pdf_file.save(pdf_path)
        print("------ SAVED PDF -----------")
#         sleep(20)
        summary, image_base64 = process_pdf(pdf_path)
        print("------ PROCESSED PDF -----------")
        
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)  # Delete the file
                print(f"{pdf_path} has been deleted successfully.")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
             print(f"{pdf_path} does not exist.")
    
        # Return the summary and Base64-encoded graph image
        return jsonify({
            "message": "PDF processed successfully.",
            "summary": summary,
            "graph": image_base64
        }), 200
    
    else:            
        if os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)  # Delete the file
                print(f"{pdf_path} has been deleted successfully.")
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
             print(f"{pdf_path} does not exist.")

        return jsonify({"message": "PDF uploaded successfully."}), 200
            
        return jsonify({"error": "No PDF file provided."}), 400
        
        
if __name__ == '__main__':
    app.run(debug=True)

