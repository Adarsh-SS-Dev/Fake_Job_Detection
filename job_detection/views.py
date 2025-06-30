from django.shortcuts import render
from .ml_models import predict_fake_job
import pandas as pd
import os

# Load dataset once for website lookup
DATASET_PATH = os.path.join(os.path.dirname(__file__), "fake_job_dataset.csv")
df = pd.read_csv(DATASET_PATH)

def index(request):
    return render(request, "job_detection/index.html")

def predict_fake_job_view(request):
    if request.method == "POST":
        try:
            # Get form data
            company_name = request.POST.get("company_name", "").strip()
            job_position = request.POST.get("job_position", "").strip()
            is_certified = request.POST.get("company_certified", "").strip()
            verified_location = request.POST.get("location_certified", "").strip()
            scam_reports = request.POST.get("scam_reports", "").strip()
            user_feedback = request.POST.get("user_feedback_score", "").strip()
            job_description = request.POST.get("job_description", "").strip()

            # Validate numeric inputs
            if not scam_reports.isdigit():
                return render(request, "job_detection/predict.html", {"error": "Scam reports must be a number."})
            if not user_feedback.replace(".", "", 1).isdigit():
                return render(request, "job_detection/predict.html", {"error": "User feedback score must be a number."})

            # Convert inputs
            scam_reports = int(scam_reports)
            user_feedback = float(user_feedback)
            verified_location = 1 if verified_location.lower() == "yes" else 0
            is_certified = 1 if is_certified.lower() == "yes" else 0

            # Lookup website from dataset
            matched_row = df[df["company_name"].str.lower() == company_name.lower()]
            if not matched_row.empty:
                company_website = matched_row.iloc[0]["company_website"]
            else:
                company_website = "Website not found"

            # Prepare input for the model
            features = {
                "company_name": company_name,
                "job_position": job_position,
                "company_certified": is_certified,
                "location_certified": verified_location,
                "user_feedback_score": user_feedback,
                "scam_reports": scam_reports,
                "job_description": job_description,
                "company_website": company_website,
            }

            # Get prediction
            prediction_result = predict_fake_job(features)
            if isinstance(prediction_result, str):  # Error handling
                return render(request, "job_detection/predict.html", {"error": prediction_result})

            result = {
                "Fake_Job": prediction_result["Fake_Job"],
                "Confidence_Score": prediction_result["Confidence_Score"],
                "Input_Values": features,
                "Company_Website": company_website,
            }

            return render(request, "job_detection/predict.html", {"result": result})

        except Exception as e:
            return render(request, "job_detection/predict.html", {"error": str(e)})

    return render(request, "job_detection/predict.html")
