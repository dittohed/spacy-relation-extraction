    import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Span
from spacy import displacy

import string

nlp = spacy.load('en_core_web_trf')
matcher = Matcher(nlp.vocab)
matcher_dep = DependencyMatcher(nlp.vocab)

# --- patterns ---
pattern1 = [
    # anchor specification
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LOWER': 'of'}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>',
        'RIGHT_ID': 'object of preposition',
        'RIGHT_ATTRS': {'DEP': 'pobj'}}
    },

]

doc = nlp('daily servings of fruit and vegetables, and fish')

displacy.serve(doc, style='dep')
