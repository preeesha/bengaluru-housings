from flask import Flask, request, jsonify
from flask_cors import CORS

from joblib import load
from model import predict, getUniqueValues

import warnings

warnings.filterwarnings(
    action="ignore",
    message=".*Unverified HTTPS.*",
)


model = load("model.joblib")


app = Flask(__name__)
CORS(app)


@app.route("/api/metadata", methods=["GET"])
def metadata():
    return jsonify(getUniqueValues())


@app.route("/api/predict", methods=["POST"])
def api():
    data = request.get_json()
    if "location" not in data:
        return jsonify({"message": "Location is required"}), 400
    if "bhk" not in data:
        return jsonify({"message": "BHK is required"}), 400
    if "bath" not in data:
        return jsonify({"message": "Bath is required"}), 400
    if "balcony" not in data:
        return jsonify({"message": "Balcony is required"}), 400
    if "sqft" not in data:
        return jsonify({"message": "Sqft is required"}), 400
    if "area_type" not in data:
        return jsonify({"message": "Area type is required"}), 400
    if "availability" not in data:
        return jsonify({"message": "Availability is required"}), 400

    price = predict(
        model,
        data["location"],
        data["bhk"],
        data["bath"],
        data["balcony"],
        data["sqft"],
        data["area_type"],
        data["availability"],
    )

    return jsonify({"price": price}), 200


if __name__ == "__main__":
    app.run(debug=True)
