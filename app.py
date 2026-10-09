"""
Flask REST API for California House Price Prediction using model.pkl
Course Outcome: CO5 - Solve real-life problems using appropriate machine learning models and evaluate performance measures.
"""

import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model.pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

with open(MODEL_PATH, "rb") as f:
    model_bundle = pickle.load(f)

pipeline = model_bundle["pipeline"]
model_name = model_bundle.get("model_name", "Gradient Boosting Regressor")
expected_features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Welcome to California House Price Prediction API",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "House Price Prediction API",
        "model_name": model_name,
        "model_loaded": True
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"success": False, "error": "Invalid JSON", "message": "JSON body required"}), 400

    missing = [f for f in expected_features if f not in data]
    if missing:
        return jsonify({"success": False, "error": "Missing features", "missing": missing}), 400

    try:
        input_data = [[float(data[f]) for f in expected_features]]
        df = pd.DataFrame(input_data, columns=expected_features)
        
        raw_pred = float(pipeline.predict(df)[0])
        clamped_pred = max(0.15, raw_pred)
        price_dollars = round(clamped_pred * 100_000, 2)

        return jsonify({
            "success": True,
            "predicted_value": round(clamped_pred, 4),
            "target_unit": "100,000 US dollars",
            "predicted_price_usd": price_dollars,
            "formatted_price": f"${price_dollars:,.2f}",
            "model_name": model_name,
            "message": "Prediction generated successfully"
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": "Prediction error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://0.0.0.0:{port} ...")
    app.run(host="0.0.0.0", port=port, debug=False)
