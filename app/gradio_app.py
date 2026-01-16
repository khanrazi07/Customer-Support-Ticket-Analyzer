import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

#for loading the models 
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")



import joblib
import numpy as np
import gradio as gr

from src.preprocessing import preprocess_text
from src.feature_engineering import (
    add_text_length,
    add_sentiment_score,
    combine_features
)
from src.entity_extraction import extract_entities


# ------------------------
# Load models & vectorizers
# ------------------------
issue_model = joblib.load(os.path.join(MODELS_DIR, "issue_model.pkl"))
urgency_model = joblib.load(os.path.join(MODELS_DIR, "urgency_model.pkl"))

issue_vectorizer = joblib.load(os.path.join(MODELS_DIR, "issue_vectorizer.pkl"))
urgency_vectorizer = joblib.load(os.path.join(MODELS_DIR, "urgency_vectorizer.pkl"))


# ------------------------
# Load product list
# ------------------------
import pandas as pd
df_products = pd.read_csv(os.path.join(DATA_DIR, "AI_tickets.csv"))
product_list = df_products["product"].dropna().unique().tolist()


# ------------------------
# Prediction pipeline
# ------------------------
def analyze_ticket(ticket_text: str):

    if not ticket_text or ticket_text.strip() == "":
        return "Invalid input", "Invalid input", {}

    # Preprocess
    processed = preprocess_text(ticket_text)

    # ----- Issue Type Prediction -----
    X_issue_tfidf = issue_vectorizer.transform([processed])
    len_issue = np.array([len(processed.split())])
    sent_issue = np.array([add_sentiment_score(pd.Series([processed]))[0]])

    X_issue = combine_features(X_issue_tfidf, len_issue, sent_issue)
    issue_pred = issue_model.predict(X_issue)[0]

    # ----- Urgency Prediction -----
    X_urg_tfidf = urgency_vectorizer.transform([processed])
    len_urg = np.array([len(processed.split())])
    sent_urg = np.array([add_sentiment_score(pd.Series([processed]))[0]])

    X_urg = combine_features(X_urg_tfidf, len_urg, sent_urg)
    urgency_pred = urgency_model.predict(X_urg)[0]

    # ----- Entity Extraction -----
    entities = extract_entities(ticket_text, product_list)

    return issue_pred, urgency_pred, entities


# ------------------------
# Gradio UI
# ------------------------
interface = gr.Interface(
    fn=analyze_ticket,
    inputs=gr.Textbox(lines=6, label="Enter Customer Support Ticket"),
    outputs=[
        gr.Textbox(label="Predicted Issue Type"),
        gr.Textbox(label="Predicted Urgency Level"),
        gr.JSON(label="Extracted Entities")
    ],
    title="Customer Support Ticket Analyzer",
    description="Predicts issue type, urgency level, and extracts key entities from customer tickets."
)

if __name__ == "__main__":
    interface.launch()
