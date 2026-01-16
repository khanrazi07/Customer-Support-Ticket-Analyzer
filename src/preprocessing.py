import re
import pandas as pd
import nltk
import spacy

from nltk.corpus import stopwords

# Download only once (safe to keep)
nltk.download("stopwords")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Perform basic text normalization:
    - lowercase
    - remove special characters and digits
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    """
    Full preprocessing pipeline:
    - cleaning
    - tokenization
    - stopword removal
    - lemmatization
    """
    if pd.isna(text):
        return ""

    text = clean_text(text)

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in STOP_WORDS and not token.is_space
    ]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()

    # Handle missing ticket text
    df[text_column] = df[text_column].fillna("")

    # Apply text preprocessing
    df["processed_text"] = df[text_column].apply(preprocess_text)

    return df


