import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)


data = joblib.load("model.pkl")  
model = data["model"]
scaler = data["scaler"]

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
  
    float_features = [float(x) for x in request.form.values()]
    features = np.array(float_features).reshape(1, -1)

  
    features = scaler.transform(features)
    prediction = model.predict(features)

    return render_template("index.html", prediction_text=f"The flower species is {prediction[0]}")

if __name__ == "__main__":
    app.run(debug=True)
