# nltk_processing.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download resources (only first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text
text = ("Relief from heat? Monsoon set to resume march.. Gradual respite from the intense heatwave in northwest India is likely from June 13 as monsoon is poised to resume its northward march after a 13-day pause since May 29, IMD and private forecaster Skymet Weather Services said on Tuesday, reports Neha Madaan.")


# Tokenization
tokens = word_tokenize(text)
print("\n🔹 Tokens:")
print(tokens)

# Stopword removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("\n🔹 After Stopword Removal:")
print(filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered_tokens]
print("\n🔹 After Stemming:")
print(stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\n🔹 After Lemmatization:")
print(lemmatized)
