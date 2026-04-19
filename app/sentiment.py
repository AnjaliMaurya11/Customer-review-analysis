import pickle
import re
import string

# =========================
# 🔹 LOAD MODEL & VECTORIZER
# =========================
model = pickle.load(open("app/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("app/tfidf_sentiment.pkl", "rb"))

# =========================
# 🔹 TEXT CLEANING
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# =========================
# 🔹 RATING → SENTIMENT
# =========================
def rating_to_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating <= 2:
        return "Negative"
    else:
        return "Neutral"

# =========================
# 🔥 ML PREDICTION (UPDATED)
# =========================
def predict_single(review, threshold=0.6):
    review = clean_text(review)
    vec = vectorizer.transform([review])

    # 🔥 Use probabilities instead of direct prediction
    proba = model.predict_proba(vec)[0]

    max_prob = max(proba)

    # 👉 Neutral if confidence is low
    if max_prob < threshold:
        return "Neutral"
    else:
        return model.classes_[proba.argmax()]

# =========================
# 🔥 MAIN FUNCTION
# =========================
def analyze_sentiment(reviews, ratings=None):
    """
    reviews: list of reviews
    ratings: optional list of ratings
    returns: sentiment percentage distribution
    """

    sentiments = []

    for i, review in enumerate(reviews):

        rating = None

        # Use rating if available
        if ratings and i < len(ratings):
            rating = ratings[i]

        if rating:
            try:
                sentiment = rating_to_sentiment(int(rating))
            except:
                sentiment = predict_single(review)
        else:
            sentiment = predict_single(review)

        sentiments.append(sentiment)

    if not sentiments:
        return {}

    total = len(sentiments)

    pos = sentiments.count("Positive")
    neg = sentiments.count("Negative")
    neu = sentiments.count("Neutral")

    return {
        "Positive %": round((pos / total) * 100, 2),
        "Negative %": round((neg / total) * 100, 2),
        "Neutral %": round((neu / total) * 100, 2)
    }