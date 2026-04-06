

from collections import Counter

from collections import Counter

def extract_keywords(reviews, top_n=10, min_freq=3):
    """
    Extract keywords that appear at least min_freq times.

    reviews: list of cleaned review strings
    top_n: max number of keywords to return
    min_freq: minimum frequency threshold
    """

    if not reviews:
        return []

    # Combine all reviews
    text = " ".join(reviews)

    # Split into words
    words = text.split()

    # Remove short words
    words = [word for word in words if len(word) > 2]

    # Count frequency
    word_counts = Counter(words)

    # 🔹 Filter words with frequency >= min_freq
    filtered_words = [(word, count) for word, count in word_counts.items() if count >= min_freq]

    # 🔹 Sort by frequency (descending)
    filtered_words = sorted(filtered_words, key=lambda x: x[1], reverse=True)

    # 🔹 Return top N
    return filtered_words[:top_n]