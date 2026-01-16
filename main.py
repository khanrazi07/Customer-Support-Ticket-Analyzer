from fastapi import FastAPI
import joblib
import os

app = FastAPI(title="Customer Support Ticket Analyzer API")

ISSUE_MODEL_PATH = os.path.join("models", "issue_model.pkl")
URGENCY_MODEL_PATH = os.path.join("models", "urgency_model.pkl")

issue_model = joblib.load(ISSUE_MODEL_PATH)
urgency_model = joblib.load(URGENCY_MODEL_PATH)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/api/predict")
def predict(issue: str):
    issue_type = issue_model.predict([issue])[0]
    urgency = urgency_model.predict([issue])[0]

    return {
        "issue": issue,
        "issue_type": issue_type,
        "urgency": urgency
    }
