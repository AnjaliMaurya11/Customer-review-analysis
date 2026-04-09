from flask import Flask, render_template, request
import pandas as pd
from app.preprocessing import preprocess_csv, preprocess_manual
from app.sentiment import analyze_sentiment
# from app.summarizer import generate_summary
from app.keywordExtract import extract_keywords  

app = Flask(__name__)


# 🔹 Upload Page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def result():
    sentiment = None
    keywords = None
    summary = None
   

    reviews = []

    file = request.files.get('file')

    # 🔹 CSV Input
    if file and file.filename != "":
        df = preprocess_csv(file)
        reviews = df["cleaned_review"].tolist()

    # 🔹 Manual Input
    else:
        manual_text = request.form.get('manual_reviews')

        if manual_text:
            reviews = preprocess_manual(manual_text)

    # 🔹 Safety check
    if not reviews:
        return "No reviews provided!"

    # 🔹 Pass cleaned reviews
    sentiment, _ = analyze_sentiment(reviews)
    keywords = extract_keywords(reviews)
    # summary = generate_summary(reviews)

    return render_template('result.html',
                           sentiment=sentiment,
                           keywords=keywords
                        #    summary=summary
                        )     


if __name__ == "__main__":
    app.run(debug=True)                                                                                        