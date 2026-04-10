@app.route("/predict", methods=["POST"])
def predict():

    amount = float(request.form["amount"])
    time = float(request.form["time"])
    freq = float(request.form["freq"])

    features = [time, amount, freq]

    scaled = scaler.transform([features])

    prediction = model.predict(scaled)

    if prediction[0] == 1:
        result = "⚠ High Risk Fraud Transaction"
    else:
        result = "✅ Legitimate Transaction"

    return render_template("index.html", prediction_text=result)