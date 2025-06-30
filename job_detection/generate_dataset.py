import pandas as pd

# Load your original dataset
df = pd.read_csv("fake_job_dataset.csv")

# Function to create structured job description
def generate_structured_jd(row):
    position = row['job_position']
    company = row['company_name']
    company_cert = "a certified company" if row['company_certified'] else "not a certified company"
    location_cert = "The job location is certified" if row['location_certified'] else "The job location is not certified"
    feedback_score = row['user_feedback_score']
    scam_reports = row['scam_reports']
    return (
        f"**Position:** {position}\n"
        f"**Company:** {company}\n\n"
        f"**Job Description:**\n"
        f"We are seeking a {position} to join our team at {company}. This role involves performing tasks and responsibilities typical for this position.\n\n"
        f"{company} is {company_cert}. {location_cert}. The position has a user feedback score of {feedback_score:.2f}, "
        f"and {scam_reports} scam report(s) have been associated with it. Applicants are advised to proceed with caution if any red flags arise."
    )

# Apply the transformation
df['job_description'] = df.apply(generate_structured_jd, axis=1)

# Save to new CSV
df.to_csv("fake_job_dataset_with_descriptions.csv", index=False)
