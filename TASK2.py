# CADL2: Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Dataset (movie reviews / tweets style)
docs = [
    "I loved the movie, it was fantastic!",
    "The movie was terrible and boring.",
    "An amazing performance by the lead actor.",
    "I didn’t like the film, it was disappointing."
]

# Bag of Words
print("🔹 Bag of Words Representation")
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(docs)
print(vectorizer.get_feature_names_out())
print(X_bow.toarray())

# TF-IDF
print("\n🔹 TF-IDF Representation")
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(docs)
print(tfidf.get_feature_names_out())
print(X_tfidf.toarray())
