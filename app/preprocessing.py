#change

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# 🔹 Clean single text
def clean_text(text):
    text = str(text).lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# 🔹 Preprocess from CSV (file OR path)
def preprocess_csv(file):
    df = pd.read_csv(file)

    print("Columns in dataset:\n", df.columns)

    if "Review Text" not in df.columns:
        raise Exception("'Review Text' column not found!")

    df["cleaned_review"] = df["Review Text"].apply(clean_text)

    return df


# 🔹 Preprocess manual input
def preprocess_manual(text):
    reviews = text.split('\n')

    cleaned = [clean_text(r) for r in reviews if r.strip() != ""]

    return cleaned