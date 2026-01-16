import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from scipy.sparse import hstack


def add_text_length(text_series):
    """
    text_series: pandas Series of processed text
    """
    return text_series.apply(lambda x: len(x.split()))


def add_sentiment_score(text_series):
    """
    text_series: pandas Series of processed text
    """
    return text_series.apply(lambda x: TextBlob(x).sentiment.polarity)


def build_tfidf_features(texts, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )
    X_tfidf = vectorizer.fit_transform(texts)
    return X_tfidf, vectorizer


def combine_features(X_tfidf, length_feat, sentiment_feat):
    numeric_features = np.vstack(
        (length_feat, sentiment_feat)
    ).T

    return hstack([X_tfidf, numeric_features])
