from preprocessing import preprocess_data
from textblob import TextBlob

# File path
file_path = "app/dataset/Womens Clothing E-Commerce Reviews.csv"

# Get preprocessed data
df = preprocess_data(file_path)

# Function to get sentiment
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
if df is not None:
    
    # Apply sentiment function
    df["sentiment"] = df["cleaned_review"].apply(get_sentiment)
    
    # Count results
    sentiment_counts = df["sentiment"].value_counts()
    
    print("\n=== SENTIMENT RESULTS ===\n")
    print(df[["Review Text", "cleaned_review", "sentiment"]].head())
    
    print("\n=== SUMMARY ===\n")
    print(sentiment_counts)

