# nltk_stemming.py

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download required tokenizer
nltk.download('punkt')

# Sample sentence
sentance= ("Relief from heat? Monsoon set to resume march.. Gradual respite from the intense heatwave in northwest India is likely from June 13 as monsoon is poised to resume its northward march after a 13-day pause since May 29, IMD and private forecaster Skymet Weather Services said on Tuesday, reports Neha Madaan.")

# Tokenize the sentence
words = word_tokenize(sentance)

# Initialize stemmer
stemmer = PorterStemmer()

# Apply stemming
stemmed_words = [stemmer.stem(word) for word in words]

# Display result
print("Original Words:")
print(words)
print("\nStemmed Words:")
print(stemmed_words)
