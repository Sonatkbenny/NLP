# spacy_lemmatization.py

import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Sample sentence
sentence= ("Relief from heat? Monsoon set to resume march.. Gradual respite from the intense heatwave in northwest India is likely from June 13 as monsoon is poised to resume its northward march after a 13-day pause since May 29, IMD and private forecaster Skymet Weather Services said on Tuesday, reports Neha Madaan.")

# Process the text
doc = nlp(sentence)

# Extract lemmas
lemmatized_words = [token.lemma_ for token in doc]

# Display result
print("Original Words:")
print([token.text for token in doc])
print("\nLemmatized Words:")
print(lemmatized_words)
