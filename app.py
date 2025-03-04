from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

app.secret_key = 'breast_cancer_prediction'

# Define paths for model and scaler
scaler_path = "model/scaler_breast.pkl"
model_path = "model/breast_cancer.pkl"

# Ensure model and scaler exist
if not os.path.exists(scaler_path) or not os.path.exists(model_path):
    raise FileNotFoundError("Model or scaler file is missing. Ensure they are in the correct directory.")

# Load the trained scaler and model
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# Define the exact order of features used during training
features_order = [
    "texture_mean", "smoothness_mean", "compactness_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean", "texture_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se", "texture_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst",
    "fractal_dimension_worst"
]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Extract input values safely
            values = []
            for feature in features_order:
                value = request.form.get(feature)  # Use `.get()` to avoid errors if key is missing
                if value is None or value.strip() == "":
                    return render_template("index.html", error=f"Missing input for {feature}")

                try:
                    values.append(float(value))
                except ValueError:
                    return render_template("index.html", error=f"Invalid value for {feature}. Please enter a valid number.")

            # Ensure all values are non-negative
            if any(val < 0 for val in values):
                return render_template("index.html", error="Values cannot be negative.")

            # Convert to numpy array and scale inputs
            values = np.array([values])
            scaled_values = scaler.transform(values)

            # Make prediction
            prediction = model.predict(scaled_values)[0]

            # Map prediction to readable output
            risk = "Malignant" if prediction == 1 else "Benign"

            return render_template("index.html", risk=risk, prediction=prediction)

        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {e}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
