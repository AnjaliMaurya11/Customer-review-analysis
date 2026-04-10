#change

from flask import Flask, render_template, request
import pandas as pd
from app.preprocessing import preprocess_csv, preprocess_manual
from app.sentiment import analyze_sentiment
from app.summarizer import generate_summary
from app.keywordExtract import extract_keywords

app = Flask(__name__)

# 🔹 Home Page
@app.route('/')
def home():
    return render_template('index.html')

# 🔹 Prediction Route
@app.route('/predict', methods=['POST'])
def result():

    sentiment = None
    keywords = None
    summary = None

    cleaned_reviews = []
    original_reviews = []

    file = request.files.get('file')

    # 🔹 CSV INPUT
    if file and file.filename != "":

        df = preprocess_csv(file)

        # cleaned text → sentiment
        cleaned_reviews = df["cleaned_review"].dropna().astype(str).tolist()

        # original text → keywords + summary
        if "Review Text" in df.columns:
            original_reviews = df["Review Text"].dropna().astype(str).tolist()
        else:
            original_reviews = cleaned_reviews

    # 🔹 MANUAL INPUT
    else:
        manual_text = request.form.get('manual_reviews')

        if manual_text:
            original_reviews = [manual_text]
            cleaned_reviews = preprocess_manual(manual_text)

    # 🔹 SAFETY CHECK
    if not cleaned_reviews:
        return "No reviews provided!"

    # 🔹 SENTIMENT → cleaned text
    sentiment, _ = analyze_sentiment(cleaned_reviews)

    # 🔥 IMPORTANT FIX (for your keywords)
    combined_text = " ".join(original_reviews)

    # 🔹 KEYWORDS → combined text
    keywords = extract_keywords(combined_text)

    # 🔹 SUMMARY → list (correct for her function)
    summary = generate_summary(original_reviews)

    print("KEYWORDS:", keywords)
    print("SUMMARY:", summary)

    # 🔹 RETURN
    return render_template(
        'result.html',
        sentiment=sentiment,
        keywords=keywords,
        summary=summary
    )

# 🔹 Run App
if __name__ == "__main__":
    app.run(debug=True)