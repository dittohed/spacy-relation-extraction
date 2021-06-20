# for quick tests (contains some examples also)
import spacy
from spacy import displacy

# --- display dependency tree example ---
nlp = spacy.load('en_core_web_sm')
doc = nlp("He suffered from cyanobacterial algal bloom-associated illness.")
displacy.serve(doc, style='dep')

# ---- display tokens' lemmas ---
# print([token.dep_ for token in doc])
