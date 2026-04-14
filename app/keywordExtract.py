import pickle
import numpy as np

vectorizer = pickle.load(open("app/tfidf.pkl", "rb"))

def extract_keywords(text_input, top_n=10):

    if not text_input or not text_input.strip():
        return []

    text_input = text_input.lower().strip()

    tfidf_matrix = vectorizer.transform([text_input])

    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    indices = np.where(scores > 0)[0]

    ranked = sorted(indices, key=lambda i: scores[i], reverse=True)

    results = []

    for i in ranked:
        phrase = feature_names[i]

        # only bigrams/trigrams
        if len(phrase.split()) < 2:
            continue

        # real occurrence count in text
        count = text_input.count(phrase)

        # skip if somehow not present
        if count == 0:
            continue

        results.append((phrase, count))

        if len(results) >= top_n:
            break

    return results