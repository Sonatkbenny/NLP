# spacy_processing.py

import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = ("Relief from heat? Monsoon set to resume march.. Gradual respite from the intense heatwave in northwest India is likely from June 13 as monsoon is poised to resume its northward march after a 13-day pause since May 29, IMD and private forecaster Skymet Weather Services said on Tuesday, reports Neha Madaan.")


# Process text
doc = nlp(text)

# Tokenization
tokens = [token.text for token in doc]
print("\n🔹 Tokens:")
print(tokens)

# Stopword removal
filtered_tokens = [token for token in doc if not token.is_stop and token.is_alpha]
print("\n🔹 After Stopword Removal:")
print([token.text for token in filtered_tokens])

# Lemmatization
lemmatized = [token.lemma_ for token in filtered_tokens]
print("\n🔹 After Lemmatization:")
print(lemmatized)

# Stemming (⚠️ Not supported in spaCy directly)
print("\n⚠️ Stemming is not supported in spaCy. Use NLTK for stemming.")
