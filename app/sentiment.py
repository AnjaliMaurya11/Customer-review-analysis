import pickle
import re
import string
import plotly.express as px
import pandas as pd


# =========================
# LOAD MODEL & VECTORIZER
# =========================
model = pickle.load(open("app/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("app/tfidf_sentiment.pkl", "rb"))


# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text


# =========================
# RATING → SENTIMENT
# =========================
def rating_to_sentiment(rating):

    if rating >= 4:
        return "Positive"

    elif rating <= 2:
        return "Negative"

    else:
        return "Neutral"


# =========================
# ML PREDICTION
# =========================
def predict_single(review, threshold=0.6):

    review = clean_text(review)

    vec = vectorizer.transform([review])

    proba = model.predict_proba(vec)[0]

    max_prob = max(proba)

    # Neutral if confidence low
    if max_prob < threshold:
        return "Neutral"

    return model.classes_[proba.argmax()]


# =========================
# SENTIMENT PIE CHART
# (VALUES ONLY ON HOVER)
# =========================
def create_sentiment_chart(pos, neg, neu):

    fig = px.pie(
        names=["Positive", "Negative", "Neutral"],
        values=[pos, neg, neu],
        title="Sentiment Distribution"
    )

    fig.update_traces(
        textinfo="none",  # hides permanent labels
        hovertemplate="<b>%{label}</b><br>Percentage: %{value}%<extra></extra>"
    )

    return fig.to_html(full_html=False)


# =========================
# RATING HISTOGRAM
# (VALUES ONLY ON HOVER)
# =========================
def create_rating_chart(ratings):

    df = pd.DataFrame(ratings, columns=["Ratings"])

    fig = px.histogram(
        df,
        x="Ratings",
        nbins=5,
        title="Rating Distribution"
    )

    fig.update_traces(
        hovertemplate="<b>Rating %{x}</b><br>Reviews: %{y}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Rating",
        yaxis_title="Number of Reviews"
    )

    return fig.to_html(full_html=False)


# =========================
# MAIN SENTIMENT FUNCTION
# =========================
def analyze_sentiment(reviews, ratings=None):

    sentiments = []
    rating_chart = None

    for i, review in enumerate(reviews):

        rating = None

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


    positive_percent = round((pos / total) * 100, 2)
    negative_percent = round((neg / total) * 100, 2)
    neutral_percent = round((neu / total) * 100, 2)


    sentiment_chart = create_sentiment_chart(
        positive_percent,
        negative_percent,
        neutral_percent
    )


    if ratings:
        rating_chart = create_rating_chart(ratings)


    return {

        "positive_percent": positive_percent,
        "negative_percent": negative_percent,
        "neutral_percent": neutral_percent,

        "sentiment_chart": sentiment_chart,
        "rating_chart": rating_chart
    }