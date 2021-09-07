# displays dependency tree for given file,
# where 2 first lines specify metadata
# (skips empty lines, see below)

import spacy
from spacy import displacy
import numpy as np

nlp = spacy.load('en_core_web_trf')
doc = nlp('I like apples and oranges.')
print(np.allclose(
    (doc[0].vector_norm + doc[1].vector_norm + doc[2].vector_norm) / 3, doc[0 : 3].vector_norm))

FILEPATH = './task 1/food/food - article 5'

nlp = spacy.load('en_core_web_trf')
with open(FILEPATH, 'r', encoding="utf-8") as file:
    lines = file.readlines()

    for line in lines[3:]:
        if line == '\n':
            continue

        doc = nlp(line)
        displacy.serve(doc, style='dep') # press ctrl+C to terminate and go to the next phrase

        # print(doc.ents)
        # for word in doc:
        #     print(word.text, word.lemma_)
        # print([token.dep_ for token in doc])
