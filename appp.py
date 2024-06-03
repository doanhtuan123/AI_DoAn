from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import the license plate detection function from the read_bien module
import read_bien

# Define server configuration
my_port = '8000'

# Initialize Flask application
app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    return "Welcome to Flask API!"

# Define route for license plate detection
@app.route('/detect_plate', methods=['POST'])
def detect_plate():
    try:
        # Get the base64 image data from the POST request
        image_b64 = request.form.get('image')
        if not image_b64:
            return jsonify({"error": "No image data provided"}), 400
        
        # Decode base64 to numpy array
        image_data = base64.b64decode(image_b64)
        np_image = np.frombuffer(image_data, dtype=np.uint8)
        
        # Decode numpy array to image
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Resize the image to 500x320
        
        
        # Call the license plate detection function
        plate_result = read_bien.detection2line_SVM(image)
        
        # Return the detection result as JSON
        return jsonify(plate_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=my_port)
