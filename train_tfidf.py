#change

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 Better training data (important!)
data = [
    "product quality is good",
    "size is small and not fitting",
    "loose threads and bad stitching",
    "nothing extraordinary about this product",
    "very comfortable and nice design",
    "delivery was fast and packaging was neat",
    "disappointed with the quality",
    "fabric is soft and comfortable",
    "poor stitching and loose threads",
    "size issue and fitting problem",
    "bad quality and not worth price",
]

# 🔹 CREATE IMPROVED VECTORIZER
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),   # ✅ THIS IS THE KEY FIX
    max_features=200
)

# 🔹 TRAIN
vectorizer.fit(data)

# 🔹 SAVE (overwrite old model)
with open("app/tfidf.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ New TF-IDF model trained successfully!")