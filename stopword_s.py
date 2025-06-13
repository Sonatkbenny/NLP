# spacy_stopword_removal.py

import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Sample sentence
sentence= ("Relief from heat? Monsoon set to resume march.. Gradual respite from the intense heatwave in northwest India is likely from June 13 as monsoon is poised to resume its northward march after a 13-day pause since May 29, IMD and private forecaster Skymet Weather Services said on Tuesday, reports Neha Madaan.")

# Process the sentence
doc = nlp(sentence)

# Remove stopwords
filtered_words = [token.text for token in doc if not token.is_stop]

# Display result
print("Original Words:")
print([token.text for token in doc])
print("\nAfter Stopword Removal:")
print(filtered_words)
