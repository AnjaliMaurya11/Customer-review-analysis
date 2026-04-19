#change

from flask import Flask, render_template, request
import pandas as pd
from app.preprocessing import preprocess_csv, preprocess_manual
from app.sentiment import analyze_sentiment
from app.summarizer import generate_summary
from app.keywordExtract import extract_keywords
from app.summarizer import generate_summary

app = Flask(__name__)

# 🔹 Home Page
@app.route('/')
def home():
    return render_template('index.html')

# 🔹 Prediction Route
@app.route('/predict', methods=['POST'])
def result():
    wordcloud_file = None
    sentiment_details = None
    keywords = None
    summary = None

    cleaned_reviews = []
    original_reviews = []
    ratings = None

    file = request.files.get('file')

    # 🔹 CSV INPUT
   
    if file and file.filename != "":

        df = preprocess_csv(file)

        # Keep alignment
        df = df.dropna(subset=["cleaned_review"])

        cleaned_reviews = df["cleaned_review"].astype(str).tolist()

        # Extract ratings if available
        if "Rating" in df.columns:
            ratings = df["Rating"].tolist()

        if "Review Text" in df.columns:
            original_reviews = df["Review Text"].dropna().astype(str).tolist()
        else:
            original_reviews = cleaned_reviews


    else:
        manual_text = request.form.get('manual_reviews')

        if manual_text:
            original_reviews = [manual_text]
            cleaned_reviews = preprocess_manual(manual_text)

    
    if not cleaned_reviews:
        return "No reviews provided!"
    
    if not original_reviews:
        original_reviews = cleaned_reviews

   
    sentiment_details = analyze_sentiment(cleaned_reviews, ratings)

    # 🔹 KEYWORDS → run RAKE per-review (works for CSV + manual input)
    keywords = extract_keywords(original_reviews)

    # 🔹 SUMMARY → list (correct for her function)
    summary = generate_summary(original_reviews, sentiment_details, keywords)

    print("KEYWORDS:", keywords)
    print("SUMMARY:", summary)

    # 🔹 RETURN
    return render_template(
        'result.html',
        sentiment_details=sentiment_details,
        keywords=keywords,
        summary=summary,
        ratings=ratings
        )

# 🔹 Run App
if __name__ == "__main__":
    app.run(debug=True)