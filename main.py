from fastapi import FastAPI, HTTPException
import joblib
import os

app = FastAPI(title="Customer Support Ticket Analyzer API")

ISSUE_MODEL_PATH = os.path.join("models", "issue_model.pkl")
URGENCY_MODEL_PATH = os.path.join("models", "urgency_model.pkl")
ISSUE_VECTOR_PATH = os.path.join("models", "issue_vectorizer.pkl")
URGENCY_VECTOR_PATH = os.path.join("models", "urgency_vectorizer.pkl")

# Load models and vectorizers
issue_model = joblib.load(ISSUE_MODEL_PATH)
urgency_model = joblib.load(URGENCY_MODEL_PATH)
issue_vectorizer = joblib.load(ISSUE_VECTOR_PATH)
urgency_vectorizer = joblib.load(URGENCY_VECTOR_PATH)

@app.get("/")
def health_check():
    return {"status": "API is running"}

from sklearn.pipeline import Pipeline

@app.post("/api/predict")
def predict(issue: str):
    try:
        issue_text = str(issue)

        # ----- Issue Type -----
        if isinstance(issue_model, Pipeline):
            issue_type = issue_model.predict([issue_text])[0]
        else:
            issue_vec = issue_vectorizer.transform([issue_text])
            issue_type = issue_model.predict(issue_vec)[0]

        # ----- Urgency -----
        if isinstance(urgency_model, Pipeline):
            urgency = urgency_model.predict([issue_text])[0]
        else:
            urgency_vec = urgency_vectorizer.transform([issue_text])
            urgency = urgency_model.predict(urgency_vec)[0]

        return {
            "issue": issue_text,
            "issue_type": issue_type,
            "urgency": urgency
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )