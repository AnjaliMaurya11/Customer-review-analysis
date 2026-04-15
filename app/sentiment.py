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
# 🔹 ML PREDICTION
# =========================
def predict_single(review):
    review = clean_text(review)
    vec = vectorizer.transform([review])
    return model.predict(vec)[0]

# =========================
# 🔥 MAIN FUNCTION (ONLY DISTRIBUTION)
# =========================
def analyze_sentiment(reviews, ratings=None):
    """
    reviews: list of cleaned reviews
    ratings: optional list of ratings
    returns: sentiment percentage distribution only
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