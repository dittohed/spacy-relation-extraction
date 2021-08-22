# for quick tests (contains some examples also)
import spacy
from spacy import displacy

# TODO: ładować plik linijka po linijce i tylko zmieniać indeks

--- display dependency tree example ---
nlp = spacy.load('en_core_web_trf')
doc = nlp('we hypothesised that eating home cooked meals more frequently would be associated with markers of a healthier diet and improved cardio-metabolic health')
# print(doc.ents)
for word in doc:
    print(word.text, word.lemma_)
displacy.serve(doc, style='dep')
# print([token.dep_ for token in doc])
