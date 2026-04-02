from preprocessing import preprocess_data

# ✅ Correct path based on your structure
file_path = "app/dataset/Womens Clothing E-Commerce Reviews.csv"

df = preprocess_data(file_path)

if df is not None:
    print("\n=== ORIGINAL REVIEWS ===\n")
    print(df["Review Text"].head())

    print("\n=== CLEANED REVIEWS ===\n")
    print(df[["Review Text", "cleaned_review"]].head())

