# Customer-review-analysis
The Customer Review Analysis & Insight Generation System is a lightweight NLP-based web application that analyzes customer reviews provided in CSV format and transforms raw textual feedback into meaningful business insights.

The system performs sentiment classification, generates a concise extractive summary, and identifies top frequent keywords from large volumes of reviews. By automating review analysis commonly collected from platforms like Amazon and Flipkart, the project helps businesses quickly understand customer satisfaction trends, common complaints, and product strengths.

The system is implemented using Python and Flask, where users upload a CSV file containing customer reviews. The file is processed using Pandas to extract review text, followed by preprocessing steps like lowercasing, stopword removal, and punctuation cleaning using NLTK. Sentiment classification is performed using TextBlob, while a frequency-based algorithm generates an extractive summary. Finally, keyword extraction is done using word frequency counting.
