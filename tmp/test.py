# displays dependency tree for given file,
# where 2 first lines specify metadata
# (skips empty lines, see below)

import spacy
from spacy import displacy
import numpy as np
import food_extractor as food
from spacy.matcher import Matcher, DependencyMatcher

nlp = spacy.load('en_core_web_lg')
doc = nlp('It was recently determined that MDS DASH eating steak regularly causes MDS flu.')
doc.ents = tuple([ent for ent in doc.ents if ent.label_ in ('DIS')])

matcher_dep = DependencyMatcher(nlp.vocab)
matcher_dep.add('dependencies', food.dependencies_patterns,
            on_match=food.add_food_dep)
matcher_dep(doc)

food.merge_entities(doc)

print(doc.ents)
for word in doc:
    print(word.text, word.lemma_, word.pos_, word.tag_, word.dep_)

for ent in doc.ents:
    print(ent, ent.label_)

displacy.serve(doc, style='dep')
