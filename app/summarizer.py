#change
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def generate_summary(reviews, top_features=6):

    if not reviews:
        return "No reviews available."

    reviews = [str(r) for r in reviews if str(r).strip() != ""]

    # Extract important keywords
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(reviews)

    feature_names = vectorizer.get_feature_names_out()
    scores = np.asarray(X.sum(axis=0)).flatten()

    ranked = scores.argsort()[::-1]

    keywords = [feature_names[i] for i in ranked[:top_features]]

    # Domain-aware clothing insights
    positive_features = []
    negative_features = []

    for word in keywords:

        if word in ["fit", "comfortable", "quality", "fabric",
                    "design", "style", "elegant", "soft",
                    "beautiful", "versatile"]:
            positive_features.append(word)

        elif word in ["size", "sizing", "large", "small",
                      "color", "tight", "loose", "length"]:
            negative_features.append(word)

    summary = ""

    if positive_features:
        summary += (
            "Customers generally appreciated the "
            + ", ".join(positive_features[:3])
            + ". "
        )

    if negative_features:
        summary += (
            "Some concerns were mentioned regarding "
            + ", ".join(negative_features[:3])
            + "."
        )

    if summary == "":
        summary = (
            "Customer feedback highlights mixed opinions "
            "with both positive experiences and some concerns."
        )

    return summary