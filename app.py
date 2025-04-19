import os
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Ensure correct paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier.h5")
EXCEL_PATH = os.path.join(BASE_DIR, "sci.xlsx")

# Load trained CNN model
model = tf.keras.models.load_model(MODEL_PATH)

# Load Excel file with plant details
df = pd.read_excel(EXCEL_PATH)

# Convert Excel data to dictionary {class_index: details}
plant_data = df.set_index("Class_Index").T.to_dict()

# Allowed file extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400

    # Ensure upload directory exists
    upload_folder = os.path.join(BASE_DIR, "static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)

    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    # Process image and predict
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    plant_details = plant_data.get(predicted_class, None)

    if not plant_details:
        plant_details = {
            "Scientific_name": "N/A",
            "Telugu_name": "N/A",
            "Parts_used": "N/A",
            "Uses": "N/A",
            "Grown_Area": "N/A",
            "Preparation_method": "N/A",
            "Preparation_method(Telugu)": "N/A"
        }
        plant_name = "Unknown"
    else:
        plant_name = plant_details["Scientific_name"]

    return render_template(
        "result.html",
        plant_name=plant_name,
        plant_details=plant_details,
        image_filename=filename
    )

if __name__ == "__main__":
    app.run(debug=True)
