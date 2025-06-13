# nltk_lemmatization.py

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download necessary data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Sample sentence
sentence= ("Relief from heat? Monsoon set to resume march.. Gradual respite from the intense heatwave in northwest India is likely from June 13 as monsoon is poised to resume its northward march after a 13-day pause since May 29, IMD and private forecaster Skymet Weather Services said on Tuesday, reports Neha Madaan.")

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Apply lemmatization (default POS is noun)
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

# Display results
print("Original Words:")
print(tokens)
print("\nLemmatized Words:")
print(lemmatized_words)
