from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pickle
import os

# Import the improved KYC extraction pipeline from cv.py
from cv import extract_aligned_kyc_features

# -------------------------------------------------
# APP SETUP
# -------------------------------------------------

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------
# LOAD MODELS (ONCE)
# -------------------------------------------------

with open("transaction_model.pkl", "rb") as f:
    transaction_model = pickle.load(f)

with open("kyc_model.pkl", "rb") as f:
    kyc_artifact = pickle.load(f)

kyc_model = kyc_artifact["model"]

with open("fusion_model.pkl", "rb") as f:
    fusion_artifact = pickle.load(f)

fusion_model = fusion_artifact["meta_model"]

# -------------------------------------------------
# FEATURE ORDER (MUST MATCH TRAINING)
# -------------------------------------------------

TRANSACTION_FEATURES = [
    "transaction_amount",
    "income",
    "txn_income_ratio",
    "amount_user_mean_ratio",
    "rolling_5_txn_amount_std",
    "txn_count_last_10_min",
    "night_txn_ratio_7day"
]

KYC_FEATURES = [
    "name_mismatch_flag",
    "invalid_id_format_flag",
    "ocr_confidence"
]

# -------------------------------------------------
# UI HOME
# -------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# -------------------------------------------------
# API (JSON – OPTIONAL)
# -------------------------------------------------

@app.route("/predict_freeze_risk", methods=["POST"])
def predict_api():

    data = request.get_json()

    txn_input = np.array(
        [data[f] for f in TRANSACTION_FEATURES]
    ).reshape(1, -1)

    transaction_risk = float(transaction_model.predict_proba(txn_input)[0, 1])

    kyc_input = np.array(
        [data[f] for f in KYC_FEATURES]
    ).reshape(1, -1)

    kyc_risk = float(kyc_model.predict_proba(kyc_input)[0, 1])

    fusion_input = np.array([[transaction_risk, kyc_risk]])
    freeze_risk = float(fusion_model.predict_proba(fusion_input)[0, 1])

    return jsonify({
        "transaction_risk_score": round(transaction_risk, 4),
        "kyc_risk_score": round(kyc_risk, 4),
        "freeze_risk_score": round(freeze_risk, 4)
    })

# -------------------------------------------------
# UI FORM HANDLER (FULL PIPELINE)
# -------------------------------------------------

@app.route("/predict_ui", methods=["POST"])
def predict_ui():

    form = request.form
    image_file = request.files["kyc_image"]
    registered_name = form["registered_name"]

    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(image_path)

    # Stage 2A — KYC document analysis (handles any image size/format)
    kyc_features = extract_aligned_kyc_features(image_path, registered_name, debug=False)

    # Stage 1
    txn_input = np.array(
        [float(form[f]) for f in TRANSACTION_FEATURES]
    ).reshape(1, -1)

    transaction_risk = float(transaction_model.predict_proba(txn_input)[0, 1])

    # Stage 2B
    kyc_input = np.array(
        [kyc_features[f] for f in KYC_FEATURES]
    ).reshape(1, -1)

    kyc_risk = float(kyc_model.predict_proba(kyc_input)[0, 1])

    # Stage 3
    fusion_input = np.array([[transaction_risk, kyc_risk]])
    freeze_risk = float(fusion_model.predict_proba(fusion_input)[0, 1])

    result = {
        "transaction_risk_score": round(transaction_risk, 4),
        "kyc_risk_score": round(kyc_risk, 4),
        "freeze_risk_score": round(freeze_risk, 4),
        **kyc_features
    }

    return jsonify(result)

# -------------------------------------------------
# RUN SERVER
# -------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
