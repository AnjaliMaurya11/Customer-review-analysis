import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")


def extract_keywords(reviews, top_n=10):

    phrase_counter = Counter()

    for review in reviews:

        doc = nlp(review)

        for chunk in doc.noun_chunks:

            phrase = chunk.text.lower().strip()

            if len(phrase.split()) >= 2:
                phrase_counter[phrase] += 1

    return phrase_counter.most_common(top_n)