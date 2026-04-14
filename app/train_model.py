import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("🚀 Training models...")

# =========================
# 🔹 LOAD DATASET
# =========================
df = pd.read_csv(
    "app/dataset/Womens Clothing E-Commerce Reviews.csv",
    encoding="latin1"
)

# Keep required columns
df = df[['Review Text', 'Rating']].dropna()

# =========================
# 🔹 CONVERT RATING → SENTIMENT
# =========================
def convert_rating(r):
    if r >= 4:
        return "Positive"
    elif r == 3:
        return "Neutral"
    else:
        return "Negative"

df['Sentiment'] = df['Rating'].apply(convert_rating)

# =========================
# 🔹 CUSTOM STOPWORDS (FIXED ✅)
# =========================
custom_stopwords = list(text.ENGLISH_STOP_WORDS - {'not', 'no', 'nor'})

# =========================
# 🔥 1. KEYWORD MODEL
# =========================
keyword_vectorizer = TfidfVectorizer(
    stop_words=custom_stopwords,
    max_features=5000,
    ngram_range=(1, 2)   # supports phrases like "not good"
)

keyword_vectorizer.fit(df['Review Text'])

# Save keyword model
pickle.dump(keyword_vectorizer, open("app/tfidf.pkl", "wb"))

# =========================
# 🔥 2. SENTIMENT MODEL
# =========================
sentiment_vectorizer = TfidfVectorizer(
    stop_words=custom_stopwords,
    max_features=5000
)

# Convert text → features
X = sentiment_vectorizer.fit_transform(df['Review Text'])
y = df['Sentiment']

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Train model
model = LogisticRegression(
    max_iter=200,
    class_weight={
        "Positive": 1,
        "Negative": 1,
        "Neutral": 2
    }
)
model.fit(X_train, y_train)

# 🔹 Evaluate
accuracy = model.score(X_test, y_test)
print("🎯 Model Accuracy:", round(accuracy * 100, 2), "%")

# 🔹 Retrain on full data (IMPORTANT)
model.fit(X, y)

# Save sentiment model
pickle.dump(sentiment_vectorizer, open("app/tfidf_sentiment.pkl", "wb"))
pickle.dump(model, open("app/sentiment_model.pkl", "wb"))

# =========================
# ✅ DONE
# =========================
print("✅ Models trained and saved successfully!")