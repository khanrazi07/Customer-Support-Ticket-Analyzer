from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sys
import pandas as pd

# ---------------- PATH SETUP (CORRECT) ----------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ---------------- INTERNAL IMPORTS ----------------
from src.preprocessing import preprocess_text
from src.entity_extraction import extract_entities
from src.feature_engineering import (
    add_text_length,
    add_sentiment_score,
    combine_features
)

# ---------------- LOAD SUPPORT DATA ----------------
DATA_FILE = os.path.join(DATA_DIR, "AI_tickets.csv")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Missing required file: {DATA_FILE}")

df_products = pd.read_csv(DATA_FILE)
product_list = df_products["product"].dropna().unique().tolist()

# ---------------- LOAD MODELS ----------------
issue_model = joblib.load(os.path.join(MODELS_DIR, "issue_model.pkl"))
urgency_model = joblib.load(os.path.join(MODELS_DIR, "urgency_model.pkl"))
issue_vectorizer = joblib.load(os.path.join(MODELS_DIR, "issue_vectorizer.pkl"))
urgency_vectorizer = joblib.load(os.path.join(MODELS_DIR, "urgency_vectorizer.pkl"))

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="Customer Support Ticket Analyzer API")

class TicketRequest(BaseModel):
    ticket_text: str

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/api/predict")
def analyze_ticket(request: TicketRequest):
    ticket_text = request.ticket_text.strip()

    if not ticket_text:
        raise HTTPException(status_code=400, detail="Ticket text cannot be empty")

    # ---------- PREPROCESS ----------
    processed = preprocess_text(ticket_text)

    # ---------- ISSUE TYPE ----------
    X_issue_tfidf = issue_vectorizer.transform([processed])
    len_issue = np.array([len(processed.split())])
    sent_issue = np.array(
        add_sentiment_score(pd.Series([processed]))
    )
    X_issue = combine_features(X_issue_tfidf, len_issue, sent_issue)
    issue_pred = issue_model.predict(X_issue)[0]

    # ---------- URGENCY ----------
    X_urg_tfidf = urgency_vectorizer.transform([processed])
    len_urg = np.array([len(processed.split())])
    sent_urg = np.array(
        add_sentiment_score(pd.Series([processed]))
    )
    X_urg = combine_features(X_urg_tfidf, len_urg, sent_urg)
    urgency_pred = urgency_model.predict(X_urg)[0]

    # ---------- ENTITY EXTRACTION ----------
    entities = extract_entities(ticket_text, product_list)

    return {
        "issue_type": issue_pred,
        "urgency": urgency_pred,
        "entities": entities
    }
