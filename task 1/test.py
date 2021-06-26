# for quick tests (contains some examples also)
import spacy
from spacy import displacy

# --- display dependency tree example ---
nlp = spacy.load('en_core_web_trf')
doc = nlp('''
bloom-associated ilness, Bartonella henselae Infection, Bird Flu, C. neoformans cryptococcosis, Crimean-Congo hemorrhagic fever
''')

displacy.serve(doc, style='dep')

# ---- display tokens' lemmas ---
# print([token.dep_ for token in doc])
