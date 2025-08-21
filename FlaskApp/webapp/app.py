from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

MODEL_PATH = "boston_housing_prediction.joblib"
# Load once at startup (faster; avoids repeated disk I/O)
clf = joblib.load(MODEL_PATH)

def scale(payload: pd.DataFrame):
    # Keeping original behavior (fit on payload) for compatibility with legacy script
    scaler = StandardScaler().fit(payload)
    return scaler.transform(payload)

@app.route("/", methods=["GET"])
def home():
    return "<h3>Sklearn Prediction Container</h3>"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Input sample:
        {
            "CHAS": { "0": 0 }, "RM": { "0": 6.575 },
            "TAX": { "0": 296 }, "PTRATIO": { "0": 15.3 },
            "B": { "0": 396.9 }, "LSTAT": { "0": 4.98 }
        }
    Output sample:
        { "prediction": [ 20.35373177134412 ] }
    """
    data = request.get_json(force=True)
    X = pd.DataFrame(data)

    try:
        preds = clf.predict(X)
    except Exception:
        # Fallback to the book’s pattern (scale on the fly)
        X_scaled = scale(X)
        preds = clf.predict(X_scaled)

    preds = [float(p) for p in np.ravel(preds)]
    return jsonify({"prediction": preds})

if __name__ == "__main__":
    # Keep debug for local use; remove/disable for production
    app.run(host="0.0.0.0", port=5000, debug=True)