from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
from PIL import Image

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "skin_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Ensure the model is compiled properly
model.compile()

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).resize((150, 150))  # Resize to (150,150,3)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array  # Shape will be (1, 150, 150, 3)

@app.route('/')
def home():
    return render_template('index.html')


# Mapping class indices to disease names
class_labels = {
    0: "Cellulitis",
    1: "Impetigo",
    2: "Dry Skin",
    3: "Athlete's Foot",
    4: "Nail Fungus",
    5: "Ringworm",
    6: "Normal Skin",
    7: "Oily Skin",
    8: "Cutaneous Larva Migrans",
    9: "Chickenpox",
    10: "Shingles"
}

# Dictionary containing disease details
disease_info = {
    "Cellulitis": {
        "Symptoms": "Redness, swelling, tenderness, stretched appearance of the skin, pain or skin sore.",
        "Possible Causes": "Bacterial infection / break in the skin / certain conditions like diabetes, lymphedema and poor circulation can increase the risk of developing cellulitis.",
        "Treatment": "Kindly consult your Dermatologist for detailed diagnosis and treatment."
    },
    "Impetigo": {
        "Symptoms": "Reddish sores or blisters that rupture, ooze and form honey-colored crusts, often around the nose and mouth.",
        "Possible Causes": "Bacterial infection / break in the skin / spread through direct skin-to-skin contact or by sharing items like towels or clothing with an infected person.",
        "Treatment": "Kindly consult your Dermatologist for detailed diagnosis and treatment."
    },
    "Athlete's Foot": {
        "Symptoms": "Itching, stinging, burning sensation, redness, flakiness, peeling skin, thickened or discolored toenails, swollen and warm feet.",
        "Possible Causes": "Fungal infection / spread through direct contact with an infected person / by touching contaminated surfaces like locker room floors, shower floors or towels.",
        "Treatment": "Kindly consult your Dermatologist for detailed diagnosis and treatment."
    },
    "Nail Fungus": {
        "Symptoms": "Nail brittleness, discoloration (white, yellow, brown or green), thickening, separation from the nail bed, debris or inflamation under the nail.",
        "Possible Causes": "Fungal infection / through infections in moist places like public pools, showers or locker rooms.",
        "Treatment": "Kindly consult your Dermatologist for detailed diagnosis and treatment."
    },
    "Ringworm": {
        "Symptoms": "Redness, swelling, itchiness, ring-shaped rash with raised, scaly borders and a clearer center.",
        "Possible Causes": "Fungal infection / spread through direct skin-to-skin contact with an infected person or animal / by touching contaminated objects like towels, clothing or gym mats.",
        "Treatment": "Kindly consult your Dermatologist for detailed diagnosis and treatment."
    },
    "Cutaneous Larva Migrans": {
        "Symptoms": "Intense itching, red, swollen lumps or blisters, winding-threadlike-raised-reddish-brown rash.",
        "Possible Causes": "Hookworm larvae (most commonly 'Dog Hookworm') / animal larvae from contaminated soil or sand.",
        "Treatment": "Kindly consult your Dermatologist for detailed diagnosis and treatment."
    },
    "Chickenpox": {
        "Symptoms": "Itchy rash with small, fluid-filled blisters that eventually scab over, often accompanied by fever and fatigue.",
        "Possible Causes": "Varicella-Zoster Virus (VZV) - through respiratory droplets from coughing or sneezing of an infected person / direct contact with the fluid from chickenpox blisters.",
        "Treatment": "Kindly consult your Dermatologist for detailed diagnosis and treatment."
    },
    "Shingles": {
        "Symptoms": "Painful rash with blisters, often on one side of the body or face, preceded by pain, tingling or burning.",
        "Possible Causes": "Reactivation of the Varicella-Zoster Virus, the same virus that causes chickenpox, Postherpetic Neuralgia (PHN).",
        "Treatment": "Kindly consult your Dermatologist for detailed diagnosis and treatment."
    },
    "Unknown Disease": {
        "Symptoms": "Information unavailable.",
        "Possible Causes": "Information unavailable.",
        "Treatment": "Information unavailable."
    },
}

@app.route('/predict', methods=['POST'])
def predict():
    print("Received request!")  # Debugging step
    if 'file' not in request.files:
        print("No file part")
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        print("Invalid file format")
        return jsonify({"error": "Invalid file format"})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    print("File saved at:", file_path)  # Debugging step

    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)

    print("Prediction Raw Output:", prediction)  # Debugging step

    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    disease_name = class_labels.get(predicted_class, "Unknown Disease")

    # Fetch disease details, default to "Unknown Disease" if not found
    disease_details = disease_info.get(disease_name, disease_info["Unknown Disease"])

    print("Predicted Class:", disease_name, "Confidence:", confidence, "Details:", disease_details)

    print("Predicted Disease Info:", disease_info.get(disease_name, {}))

    return jsonify({
        "class": int(predicted_class),
        "disease": disease_name,
        "confidence": float(confidence),
        "details": disease_details,  # Now passing disease details properly
        "file_path": f"/uploads/{filename}"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

