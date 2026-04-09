#changes

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

print("🚀 Training TF-IDF model...")

# Load dataset
df = pd.read_csv("app/dataset/Womens Clothing E-Commerce Reviews.csv", encoding='latin1')

# Keep only text column
df = df[['Review Text']]
df = df.dropna()

texts = df['Review Text']

# Train TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
vectorizer.fit(texts)

# Save model
pickle.dump(vectorizer, open("app/tfidf.pkl", "wb"))

print("✅ Model saved as tfidf.pkl")