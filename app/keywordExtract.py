

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# ✅ Load trained TF-IDF model safely
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "tfidf.pkl")

vectorizer = pickle.load(open(model_path, "rb"))

# ⚠️ IMPORTANT: Upgrade vectorizer to support phrases (bigrams)
vectorizer.max_features = 100     # optional: allows more keywords


def extract_keywords(text, top_n=10):
    # 🔹 Transform text
    tfidf_matrix = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    # 🔹 Get top scoring indices
    top_indices = scores.argsort()[-top_n:][::-1]

    keywords = []

    for i in top_indices:
        if scores[i] > 0.1:
            keywords.append(feature_names[i])

    # 🚨 FALLBACK
    if not keywords:
        words = text.split()
        keywords = words[:top_n]

    # ✅ FIXED COUNTING (works for phrases too)
    keyword_with_count = []
    for word in keywords:
        count = text.lower().count(word.lower())
        keyword_with_count.append((word, count))

    return keyword_with_count