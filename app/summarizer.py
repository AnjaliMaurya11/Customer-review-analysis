import os

def generate_summary(reviews, sentiment_result, keywords):
    """
    reviews: list of review texts
    sentiment_result: dict from analyze_sentiment()
    keywords: list of tuples from extract_keywords()
              e.g. [("great fit", 10), ("poor quality", 5)]
    """

    if not reviews:
        return "No reviews available."

    summary_parts = []

    # =========================
    # 🔹 SENTIMENT INSIGHT
    # =========================
    pos = sentiment_result.get("Positive %", 0)
    neg = sentiment_result.get("Negative %", 0)
    neu = sentiment_result.get("Neutral %", 0)

    if pos > 60:
        summary_parts.append(
            "Overall customer sentiment is strongly positive."
        )
    elif neg > 40:
        summary_parts.append(
            "Customer sentiment shows significant dissatisfaction."
        )
    else:
        summary_parts.append(
            "Customer feedback reflects a mix of positive and negative experiences."
        )

    # =========================
    # 🔹 KEYWORD INSIGHT
    # =========================
    positive_words = []
    negative_words = []

    negative_indicators = [
        "not", "poor", "bad", "small", "large",
        "tight", "loose", "issue", "problem", "short"
    ]

    for phrase, count in keywords:

        # classify based on negative words
        if any(neg_word in phrase for neg_word in negative_indicators):
            negative_words.append(phrase)
        else:
            positive_words.append(phrase)

    # =========================
    # 🔹 BUILD SUMMARY
    # =========================

    if positive_words:
        summary_parts.append(
            "Customers frequently appreciated aspects such as "
            + ", ".join(positive_words[:3])
            + "."
        )

    if negative_words:
        summary_parts.append(
            "However, some concerns were raised regarding "
            + ", ".join(negative_words[:3])
            + "."
        )

    # =========================
    # 🔹 FALLBACK
    # =========================
    if len(summary_parts) == 1:
        summary_parts.append(
            "Key themes were identified from customer reviews."
        )

    return " ".join(summary_parts)
