from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample Documents (corrected)
docs = [
    "Natural language processing enables computers to understand human language.",  
    "Chatbots use NLP to communicate with users effectively.",   
    "Text summarization helps in extracting key points from large documents.", 
    "Machine translation converts text from one language to another using AI."
]

# 1. TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(docs)

# 2. Print TF-IDF Matrix (terminal output)
print("\n🔍 TF-IDF Matrix:")
tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_df)

# 3. Word Cloud for Each Document
words = tfidf_vectorizer.get_feature_names_out()
fig, axs = plt.subplots(1, len(docs), figsize=(18, 4))

for i in range(len(docs)):
    tfidf_scores = X_tfidf[i].toarray().flatten()
    tfidf_dict = dict(zip(words, tfidf_scores))
    wordcloud = WordCloud(background_color='white').generate_from_frequencies(tfidf_dict)
    
    axs[i].imshow(wordcloud, interpolation='bilinear')
    axs[i].axis('off')
    axs[i].set_title(f'Doc {i+1}')

plt.suptitle("TF-IDF Word Clouds for Each Document")
plt.tight_layout()
plt.show()

# 4. Cosine Similarity Matrix
similarity_matrix = cosine_similarity(X_tfidf)

# 5. Print Similarity Matrix (terminal output)
print("\n🔗 Cosine Similarity Matrix:")
sim_df = pd.DataFrame(similarity_matrix, 
                      columns=[f'Doc {i+1}' for i in range(len(docs))],
                      index=[f'Doc {i+1}' for i in range(len(docs))])
print(sim_df.round(2))

# 6. Show Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(sim_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Document Similarity Matrix (TF-IDF)")
plt.show()