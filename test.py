# for quick tests (contains some examples also)
import spacy
from spacy import displacy

# --- display dependency tree example ---
nlp = spacy.load('en_core_web_trf')
doc = nlp('studies assessing relationships between diet and depression')
# print(doc.ents)
for word in doc:
    print(word.text, word.lemma_)
displacy.serve(doc, style='dep')
# print([token.dep_ for token in doc])
