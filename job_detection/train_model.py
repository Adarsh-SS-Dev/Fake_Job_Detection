import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
import joblib

# Constants
DATASET_PATH = "fake_job_dataset.csv"
MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Initialize encoders
encoder_company = LabelEncoder()
encoder_position = LabelEncoder()
encoder_certified = LabelEncoder()

# Encode categorical fields
df["company_name"] = encoder_company.fit_transform(df["company_name"])
df["job_position"] = encoder_position.fit_transform(df["job_position"])
df["company_certified"] = encoder_certified.fit_transform(df["company_certified"])
df["location_certified"] = df["location_certified"].astype(int)
df["company_website"] = df["company_website"].fillna("unknown").astype(str)
df["job_description"] = df["job_description"].fillna("")

# Define features and target
features = [
    "company_name", "job_position", "company_certified",
    "location_certified", "user_feedback_score", "scam_reports",
    "job_description", "company_website"
]
X = df[features]
y = df["is_fake"]

# Text preprocessing with TF-IDF
preprocessor = ColumnTransformer(transformers=[
    ("desc", TfidfVectorizer(max_features=100), "job_description"),
    ("website", TfidfVectorizer(max_features=20), "company_website")
], remainder='passthrough')

# Transform features
X_transformed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and preprocessor
joblib.dump(model, os.path.join(MODEL_DIR, "fake_job_model.pkl"))
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "feature_preprocessor.pkl"))


print("✅ Model and encoders trained and saved successfully!")
