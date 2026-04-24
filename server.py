from flask import Flask, request, jsonify
from flask_cors import CORS  
import util
import sys

sys.modules['__main__'].flatten_images = util.flatten_images

app = Flask(__name__)
CORS(app) 

@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        image_data = request.form['image_data']
        result = util.classify_image(image_data)
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}") 
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    util.load_saved_artifacts()
    app.run(port=5000)