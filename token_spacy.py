import spacy

nlp = spacy.load("en_core_web_sm")

sentence= nlp("Relief from heat? Monsoon set to resume march.. Gradual respite from the intense heatwave in northwest India is likely from June 13 as monsoon is poised to resume its northward march after a 13-day pause since May 29, IMD and private forecaster Skymet Weather Services said on Tuesday, reports Neha Madaan.")

for token in sentence:
    print(token.text)