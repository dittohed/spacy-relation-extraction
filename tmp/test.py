# displays dependency tree for given file,
# where 2 first lines specify metadata
# (skips empty lines, see below)

import spacy
from spacy import displacy
import numpy as np

nlp = spacy.load('en_core_web_trf')
doc = nlp('consumption of fruit, vegetables, sweets / chocolate and sugar-containing soft drinks')

print(doc.ents)
for word in doc:
    print(word.text, word.lemma_, word.pos_, word.tag_, word.dep_)

displacy.serve(doc, style='dep')
