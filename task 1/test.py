# for quick tests (contains some examples also)
import spacy
from spacy import displacy

# --- display dependency tree example ---
nlp = spacy.load('en_core_web_trf')
doc = nlp('AIDS and ALS are among most dangerous diseases discovered by Apple and UN users')
print(doc.ents)

displacy.serve(doc, style='ent')

# ---- display tokens' lemmas ---
# print([token.dep_ for token in doc])
