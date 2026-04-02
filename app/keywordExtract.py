import pandas as pd
from collections import Counter


def extract_keywords(top_n=10):
    input_path = "app/dataset/cleaned_reviews.csv"

    # Load cleaned dataset
    df = pd.read_csv(input_path)

    # Safety check
    if 'cleaned_review' not in df.columns:
        raise Exception("Missing 'cleaned_review' column. Run preprocessing first.")

    # Combine all cleaned reviews into one string
    text = " ".join(df['cleaned_review'].dropna())

    # Split into words
    words = text.split()

    # Remove very short words (optional but improves quality)
    words = [word for word in words if len(word) > 2]

    # Count word frequency
    word_counts = Counter(words)

    # Get top N keywords
    top_keywords = word_counts.most_common(top_n)

    return top_keywords


# 🔹 Run independently (for testing)
if __name__ == "__main__":
    keywords = extract_keywords(10)

    print("Top Keywords:\n")
    for word, count in keywords:
        print(f"{word} : {count}")