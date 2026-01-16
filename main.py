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

@app.post("/api/predict")
def predict(issue: str):
    try:
        issue_text = str(issue)

        # Vectorize input
        issue_vec = issue_vectorizer.transform([issue_text])
        urgency_vec = urgency_vectorizer.transform([issue_text])

        # Predict using VECTORS (THIS WAS THE FIX)
        issue_type = issue_model.predict(issue_vec)[0]
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
