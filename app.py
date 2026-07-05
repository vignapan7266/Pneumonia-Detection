from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Model
model = tf.keras.models.load_model("pneumonia_densenet121.keras")

IMG_SIZE = (224, 224)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=IMG_SIZE)
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction >= 0.5:
        result = "PNEUMONIA"
        confidence = prediction * 100
    else:
        result = "NORMAL"
        confidence = (1 - prediction) * 100

    confidence = round(confidence, 2)

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        image=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)