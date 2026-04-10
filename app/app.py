from flask import Flask, render_template, request
import joblib
import numpy as np
from waitress import serve

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    amount = float(request.form["amount"])
    time = float(request.form["time"])
    freq = float(request.form["freq"])

    features = [time, amount, freq]

    prediction = model.predict([features])

    if prediction[0] == 1:
        result = "⚠ High Risk Fraud Transaction"
    else:
        result = "✅ Legitimate Transaction"

    return render_template(
        "index.html",
        prediction_text=result
    )


# Start server with Waitress
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)