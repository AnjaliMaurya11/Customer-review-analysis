from flask import Flask, render_template, request
import pandas as pd

from app.preprocessing import preprocess_csv, preprocess_manual
from app.sentiment import analyze_sentiment
from app.keywordExtract import extract_keywords
from app.summarizer import generate_summary

app = Flask(__name__)


# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template('index.html')


# =========================
# RESULT DASHBOARD PAGE
# =========================
@app.route('/predict', methods=['POST'])
def result():

    cleaned_reviews = []
    original_reviews = []
    ratings = None

    file = request.files.get('file')


    # =========================
    # CSV INPUT
    # =========================
    if file and file.filename != "":

        df = preprocess_csv(file)

        df = df.dropna(subset=["cleaned_review"])

        cleaned_reviews = df["cleaned_review"].astype(str).tolist()


        # Ratings column optional
        if "Rating" in df.columns:
            ratings = df["Rating"].dropna().tolist()


        # Original review column optional
        if "Review Text" in df.columns:
            original_reviews = df["Review Text"].dropna().astype(str).tolist()
        else:
            original_reviews = cleaned_reviews


    # =========================
    # MANUAL INPUT
    # =========================
    else:

        manual_text = request.form.get('manual_reviews')

        if manual_text:
            original_reviews = [manual_text]
            cleaned_reviews = preprocess_manual(manual_text)


    # =========================
    # VALIDATION
    # =========================
    if not cleaned_reviews:
        return "No reviews provided!"


    if not original_reviews:
        original_reviews = cleaned_reviews


    # =========================
    # SENTIMENT ANALYSIS
    # =========================
    sentiment_details = analyze_sentiment(
        cleaned_reviews,
        ratings if ratings else None
    )


    # =========================
    # KEYWORD EXTRACTION
    # =========================
    keywords = extract_keywords(original_reviews)


    # =========================
    # SUMMARY GENERATION
    # =========================
    summary = generate_summary(
        original_reviews,
        sentiment_details,
        keywords
    )


    # =========================
    # SEND DATA TO DASHBOARD
    # =========================
    return render_template(

        "result.html",

        positive_percent=sentiment_details.get("positive_percent", 0),
        negative_percent=sentiment_details.get("negative_percent", 0),
        neutral_percent=sentiment_details.get("neutral_percent", 0),

        sentiment_chart=sentiment_details.get("sentiment_chart"),
        rating_chart=sentiment_details.get("rating_chart"),

        keywords=keywords,
        summary=summary
    )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)