import os
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
MODEL_DIR = "ml_models"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_job_model.pkl")
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(os.path.join(MODEL_DIR, "feature_preprocessor.pkl"))

# Load encoders for categorical fields
encoder_company = joblib.load(os.path.join(MODEL_DIR, "label_encoder_company.pkl"))
encoder_job = joblib.load(os.path.join(MODEL_DIR, "label_encoder_position.pkl"))
encoder_certified = joblib.load(os.path.join(MODEL_DIR, "label_encoder_certified.pkl"))

  # Make sure this is at the top

def safe_encode(encoder, value):
    classes = encoder.classes_
    if value not in classes:
        # Add the new value to classes safely
        classes = np.append(classes, value)
        encoder.classes_ = classes
    return encoder.transform([value])[0]



def predict_fake_job(data):
    """Predict if a job post is fake or legit."""

    required_fields = [
        "company_name", "job_position", "company_certified", "location_certified",
        "user_feedback_score", "scam_reports", "job_description", "company_website"
    ]
    for field in required_fields:
        if data.get(field) in [None, ""]:
            return f"Prediction Error: Missing value for '{field}'."

    # Safe encode categorical values
    company_encoded = safe_encode(encoder_company, data["company_name"])
    job_encoded = safe_encode(encoder_job, data["job_position"])
    cert_encoded = safe_encode(encoder_certified, data["company_certified"])

    # Build input DataFrame
    input_df = pd.DataFrame([{
        "company_name": company_encoded,
        "job_position": job_encoded,
        "company_certified": cert_encoded,
        "location_certified": int(data["location_certified"]),
        "user_feedback_score": float(data["user_feedback_score"]),
        "scam_reports": float(data["scam_reports"]),
        "job_description": data["job_description"],
        "company_website": data["company_website"]
    }])

    # Transform features
    X_transformed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(X_transformed)[0]
    probability = model.predict_proba(X_transformed)[0]
    confidence_score = max(probability) * 100

    return {
        "Fake_Job": "Fake" if prediction else "Legit",
        "Confidence_Score": f"{confidence_score:.2f}%"
    }
