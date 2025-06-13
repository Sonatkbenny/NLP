# nltk_stopword_removal.py

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample sentence
sentence= ("Relief from heat? Monsoon set to resume march.. Gradual respite from the intense heatwave in northwest India is likely from June 13 as monsoon is poised to resume its northward march after a 13-day pause since May 29, IMD and private forecaster Skymet Weather Services said on Tuesday, reports Neha Madaan.")

# Tokenize the sentence
words = word_tokenize(sentence)

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Remove stopwords
filtered_words = [word for word in words if word.lower() not in stop_words]

# Display result
print("Original Words:")
print(words)
print("\nAfter Stopword Removal:")
print(filtered_words)
