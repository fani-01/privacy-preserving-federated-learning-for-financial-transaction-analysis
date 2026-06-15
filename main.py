from flask import Flask, render_template, request, redirect, url_for
from model import load_federated_model, predict_fraud
import joblib

app = Flask(__name__)

# ------------------ Config ------------------
CATEGORICAL_COLS = [
    "Gender",
    "State",
    "City",
    "Bank",
    "Account_Type",
    "Transaction_Type",
    "Merchant_Category",
    "Transaction_Device",
    "Device_Type",
]

NUMERIC_COLS = ["Age", "Transaction_Amount", "Account_Balance"]

INPUT_SIZE = 12
MODEL_PATH = "federated_model.pth"

# ------------------ Load Model & Encoders ------------------
model = load_federated_model(MODEL_PATH, INPUT_SIZE)

try:
    label_encoders = joblib.load("label_encoders.pkl")
    CATEGORY_OPTIONS = {
        col: list(enumerate(label_encoders[col].classes_))
        for col in CATEGORICAL_COLS
        if col in label_encoders
    }
except Exception:
    # If encoders are missing, fall back to empty options
    CATEGORY_OPTIONS = {col: [] for col in CATEGORICAL_COLS}


# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        values = []

        # Categorical (label-encoded via dropdown values)
        for col in CATEGORICAL_COLS:
            values.append(float(request.form[col]))

        # Numeric
        for col in NUMERIC_COLS:
            values.append(float(request.form[col]))

        prediction, confidence = predict_fraud(model, values)

        return render_template(
            "result.html",
            prediction=prediction,
            confidence=confidence,
        )

    return render_template(
        "predict.html",
        categorical_cols=CATEGORICAL_COLS,
        numeric_cols=NUMERIC_COLS,
        category_options=CATEGORY_OPTIONS,
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
