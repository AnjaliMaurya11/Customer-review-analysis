#change
from textblob import TextBlob

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


def analyze_sentiment(reviews):
    """
    reviews: list of cleaned review strings
    """

    if not reviews:
        return {}, []

    sentiment_list = []

    # 🔹 Get sentiment for each review
    for review in reviews:
        sentiment = get_sentiment(review)
        sentiment_list.append(sentiment)

    # 🔹 Count percentages
    total = len(sentiment_list)

    result = {
        "Positive": round((sentiment_list.count("Positive") / total) * 100, 2),
        "Negative": round((sentiment_list.count("Negative") / total) * 100, 2),
        "Neutral": round((sentiment_list.count("Neutral") / total) * 100, 2)
    }

    return result, sentiment_list