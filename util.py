import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
from sklearn.preprocessing import FunctionTransformer # Required for the model pipeline

# This function must be defined here for the pickle loader to find it
def flatten_images(x):
    return x.reshape(x.shape[0], -1)

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def load_saved_artifacts():
    print("loading artifacts...")
    global __class_name_to_number
    global __class_number_to_name
    global __model

    # Use relative paths or verify these absolute paths exist on your machine
    dict_path = r"dict_path"
    model_path = r"model_path"

    with open(dict_path, "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    if __model is None:
        __model = joblib.load(model_path)
    print("artifacts loaded")

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    cropped = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped.append(roi_color)
    return cropped

def classify_image(image_base64_data=None, file_path=None):
    img = get_cv2_image_from_base64_string(image_base64_data) if image_base64_data else cv2.imread(file_path)
    faces = get_cropped_image_if_2_eyes(img)
    
    results = []
    for face in faces:
        scalled_raw_img = cv2.resize(face, (32, 32))
        img_har = w2d(face, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))
        final = combined.reshape(1, -1).astype(float)

        # --- THE FIX IS HERE ---
        # This will now work without crashing because of probability=True
        prediction = __model.predict(final)[0]
        probability = __model.predict_proba(final)[0] 
        
        results.append({
            "class": __class_number_to_name[prediction],
            # Multiplied by 100 to get percentage (e.g., 0.85 -> 85.0)
            "class_probability": np.round(probability * 100, 2).tolist(),
            "class_dictionary": __class_name_to_number
        })
    return results
