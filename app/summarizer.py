from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def generate_summary(reviews, num_sentences=5):

    if not reviews:
        return "No reviews available for summary."

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(reviews)

    scores = np.array(X.sum(axis=1)).flatten()

    top_indices = scores.argsort()[-num_sentences:][::-1]

    summary_sentences = [reviews[i] for i in top_indices]

    summary = " ".join(summary_sentences)

    return summary