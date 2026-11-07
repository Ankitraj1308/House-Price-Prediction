"""
app.py - Flask Web Application for House Price Prediction
==========================================================
Provides web interface and REST API to predict house prices using a trained
Gradient Boosting Regressor model pipeline on the Ames Housing dataset.
"""

import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "house-price-predictor-secret-key-2026")

# Global paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")

# Load model and metrics at startup
model_pipeline = None
model_metadata = None

try:
    if os.path.exists(MODEL_PATH):
        model_pipeline = joblib.load(MODEL_PATH)
        print("✓ Trained model pipeline loaded successfully.")
    else:
        print("⚠ Warning: models/best_model.joblib not found. Run train.py first.")
except Exception as e:
    print(f"Error loading model pipeline: {e}")

try:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            model_metadata = json.load(f)
        print("✓ Model metrics metadata loaded successfully.")
    else:
        print("⚠ Warning: models/model_metrics.json not found.")
except Exception as e:
    print(f"Error loading model metrics: {e}")

# Fallback neighborhoods if metadata not loaded
DEFAULT_NEIGHBORHOODS = [
    'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
    'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
    'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown',
    'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'
]

USD_TO_INR_RATE = 83.5


def format_inr(amount):
    """Format numeric amount into Indian numbering system (Lakhs / Crores)."""
    try:
        val = float(amount)
        if val >= 10000000:
            return f"₹ {val / 10000000:.2f} Cr"
        elif val >= 100000:
            return f"₹ {val / 100000:.2f} Lakh"
        else:
            return f"₹ {val:,.0f}"
    except Exception:
        return f"₹ {amount}"


@app.template_filter('currency_usd')
def currency_usd_filter(val):
    try:
        return f"${float(val):,.0f}"
    except Exception:
        return str(val)


@app.template_filter('currency_inr')
def currency_inr_filter(val):
    return format_inr(val)


@app.route("/")
def index():
    """Home landing page."""
    return render_template(
        "index.html",
        metadata=model_metadata
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Prediction form and submission handler."""
    neighborhoods = (
        model_metadata['dataset_summary']['neighborhoods']
        if model_metadata and 'dataset_summary' in model_metadata
        else DEFAULT_NEIGHBORHOODS
    )

    if request.method == "GET":
        return render_template("predict.html", neighborhoods=neighborhoods)

    # Handle POST submission
    try:
        if model_pipeline is None:
            flash("Model pipeline is not loaded. Please train the model first.", "danger")
            return redirect(url_for("predict"))

        # Extract and parse inputs with validation
        overall_qual = int(request.form.get("overall_qual", 6))
        overall_cond = int(request.form.get("overall_cond", 5))
        gr_liv_area = float(request.form.get("gr_liv_area", 1500))
        lot_area = float(request.form.get("lot_area", 9500))
        total_bsmt_sf = float(request.form.get("total_bsmt_sf", 900))
        bedroom_abv_gr = int(request.form.get("bedroom_abv_gr", 3))
        full_bath = int(request.form.get("full_bath", 2))
        floors = float(request.form.get("floors", 1.0))
        garage_cars = int(request.form.get("garage_cars", 2))
        year_built = int(request.form.get("year_built", 2000))
        neighborhood = request.form.get("neighborhood", "CollgCr")

        # Range validation
        if not (1 <= overall_qual <= 10):
            raise ValueError("Overall Quality must be between 1 and 10.")
        if not (1 <= overall_cond <= 10):
            raise ValueError("Overall Condition must be between 1 and 10.")
        if gr_liv_area <= 100 or gr_liv_area > 15000:
            raise ValueError("Living area must be between 100 and 15,000 sq ft.")
        if lot_area <= 100:
            raise ValueError("Lot area must be greater than 100 sq ft.")
        if year_built < 1850 or year_built > 2030:
            raise ValueError("Year built must be between 1850 and 2030.")

        # Construct single-row DataFrame matching the exact training feature names
        input_dict = {
            'OverallQual': [overall_qual],
            'GrLivArea': [gr_liv_area],
            'BedroomAbvGr': [bedroom_abv_gr],
            'FullBath': [full_bath],
            'Floors': [floors],
            'GarageCars': [garage_cars],
            'YearBuilt': [year_built],
            'TotalBsmtSF': [total_bsmt_sf],
            'LotArea': [lot_area],
            'OverallCond': [overall_cond],
            'Neighborhood': [neighborhood]
        }
        input_df = pd.DataFrame(input_dict)

        # Make prediction
        predicted_usd = float(model_pipeline.predict(input_df)[0])
        # Safety bound (ensure non-negative)
        predicted_usd = max(10000.0, round(predicted_usd, 2))
        predicted_inr = round(predicted_usd * USD_TO_INR_RATE, 2)

        input_summary = {
            "Overall Quality": f"{overall_qual} / 10",
            "Overall Condition": f"{overall_cond} / 10",
            "Living Area": f"{gr_liv_area:,.0f} sq ft",
            "Lot Area": f"{lot_area:,.0f} sq ft",
            "Total Basement Area": f"{total_bsmt_sf:,.0f} sq ft",
            "Bedrooms": f"{bedroom_abv_gr}",
            "Bathrooms": f"{full_bath}",
            "Floors": f"{floors} Stories",
            "Garage Capacity": f"{garage_cars} Cars",
            "Year Built": f"{year_built}",
            "Neighborhood": neighborhood
        }

        return render_template(
            "result.html",
            predicted_usd=predicted_usd,
            predicted_inr=predicted_inr,
            formatted_inr=format_inr(predicted_inr),
            usd_rate=USD_TO_INR_RATE,
            input_summary=input_summary,
            best_model_name=model_metadata.get("best_model_name", "Gradient Boosting Regressor") if model_metadata else "Machine Learning Regressor"
        )

    except Exception as e:
        flash(f"Prediction Error: {str(e)}", "danger")
        return redirect(url_for("predict"))


@app.route("/dashboard")
def dashboard():
    """Model analytics and evaluation dashboard."""
    return render_template(
        "dashboard.html",
        metadata=model_metadata
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic inference."""
    if model_pipeline is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503

    try:
        data = request.get_json(force=True)
        required_fields = [
            'OverallQual', 'GrLivArea', 'BedroomAbvGr', 'FullBath',
            'Floors', 'GarageCars', 'YearBuilt', 'TotalBsmtSF',
            'LotArea', 'OverallCond', 'Neighborhood'
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing required field: {field}"}), 400

        input_df = pd.DataFrame([{f: data[f] for f in required_fields}])
        pred_usd = float(model_pipeline.predict(input_df)[0])
        pred_usd = max(10000.0, round(pred_usd, 2))
        pred_inr = round(pred_usd * USD_TO_INR_RATE, 2)

        return jsonify({
            "status": "success",
            "predicted_price_usd": pred_usd,
            "predicted_price_inr": pred_inr,
            "formatted_inr": format_inr(pred_inr),
            "exchange_rate": USD_TO_INR_RATE,
            "model_used": model_metadata.get("best_model_name", "Gradient Boosting Regressor") if model_metadata else "Gradient Boosting Regressor"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    import socket

    # Determine port: use PORT env var, or try 5000, falling back to 5050 if 5000 is in use (macOS AirPlay)
    port = int(os.environ.get("PORT", 0))
    if port == 0:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 5000))
                port = 5000
        except OSError:
            print("Port 5000 is in use (common with macOS AirPlay Receiver). Falling back to port 5050.")
            port = 5050

    print(f"Starting House Price Prediction application on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
