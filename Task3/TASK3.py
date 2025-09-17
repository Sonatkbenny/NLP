# CADL3: Named Entity Recognition
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_sm")

# Example dataset: job postings
text = """
OpenAI is hiring Machine Learning Engineers in San Francisco.
Google announced a new AI research center in London.
Dr. Smith from MIT collaborated with Microsoft Research.
"""

doc = nlp(text)

# Extract named entities
print("🔹 Named Entities")
for ent in doc.ents:
    print(ent.text, ent.label_)

# Structured Information (Persons and Organizations)
entities = [(ent.text, ent.label_) for ent in doc.ents]
df = pd.DataFrame(entities, columns=["Entity", "Type"])
print("\nStructured Info:\n", df)
