import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (only first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    text = str(text).lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize words
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

# Main preprocessing function
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    print("Columns in dataset:\n", df.columns)
    
    # Extract review column (IMPORTANT)
    if "Review Text" in df.columns:
        reviews = df["Review Text"]
        
        # Clean reviews
        df["cleaned_review"] = reviews.apply(clean_text)
        
    else:
        print("❌ 'Review Text' column not found!")
        return None
    
    return df