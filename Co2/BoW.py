from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Input documents
documents = [
    "Artificial intelligence is the future.",
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning is a part of machine learning.",
    "The future is powered by artificial intelligence."
]

# Bag-of-Words vectorization
vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
df_bow = pd.DataFrame(bow_matrix.toarray(), columns=feature_names)

# Generate and save/display each word cloud separately
for idx, row in df_bow.iterrows():
    bow_counts = row.to_dict()
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(bow_counts)
    
    # Display each in a separate figure
    plt.figure(figsize=(6, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"BoW Word Cloud - Document {idx + 1}")
    plt.tight_layout()
    
    # Save as individual image
    plt.savefig(f"bow_wordcloud_doc{idx + 1}.png")
    plt.show()
