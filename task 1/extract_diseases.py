# TODO: detecting initialisms (e.g. AIDS) & standalones (e.g. anemia)
# for now, the text is scanned quadruple times (each pass for each pattern)

import spacy
from spacy.matcher import DependencyMatcher
from spacy.tokens import Span
from spacy import displacy

import string

nlp = spacy.load('en_core_web_trf')
matcher = DependencyMatcher(nlp.vocab)

# --- keywords ---
keywords = [
    'infection',
    'syndrome',
    'disorder',
    'deficit',
    'fever',
    'disease',
    'cancer',
    'flu',
    'diarrhea',
    'cold',
    'poisoning',
    'defect',
    'ilness'
]

keywords_regexp = '.*(is|us|ism|ysm|virus|pathy|pox|ia)$'

# --- modifiers ---
modifier = {
    'LEFT_ID': 'anchor',
    'REL_OP': '>',
    'RIGHT_ID': 'modifier',
    'RIGHT_ATTRS': {'DEP': {'IN': ['amod', 'compound', 'poss', 'nmod', 'npadvmod']}}
}

modmodifier = {
    'LEFT_ID': 'modifier',
    'REL_OP': '>',
    'RIGHT_ID': 'modmodifier',
    'RIGHT_ATTRS': {'DEP': {'IN': ['amod', 'compound', 'poss', 'nmod', 'npadvmod']}}
}

# --- patterns ---
pattern1 = [
    # anchor specification
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LOWER': {'IN': keywords}}
    },
    # modifier specification
    modifier
]

pattern2 = [
    # anchor specification
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LOWER': {'IN': keywords}}
    },
    modifier, # modifier specification
    modmodifier # modifier's modifiers specification
]

pattern3 = [
    # anchor specification
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LOWER': {'REGEX': keywords_regexp}}
    },
    modifier
]

pattern4 = [
    # anchor specification
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LOWER': {'REGEX': keywords_regexp}}
    },
    modifier, # modifier specification
    modmodifier # modifier's modifiers specification
]

def add_disease_ent(matcher, doc, i, matches):
    '''
    Creates entity label for current match.
    '''
    match_id, token_ids = matches[i]
    start = min(token_ids)
    end = max(token_ids) + 1

    entity = Span(doc, start, end, label='DIS')
    print(f'Matched text: {entity.text}')

    try:
        doc.ents += (entity,)
    except ValueError:
        pass # Span simply won't be added

matcher.add('disease_patterns', [pattern2, pattern4,  pattern1, pattern3],
            on_match=add_disease_ent)

s = '''
bloom-associated ilness, Bartonella henselae Infection, Bird Flu, C. neoformans cryptococcosis, Crimean-Congo hemorrhagic fever
'''
# s = s.translate(s.maketrans('', '', string.punctuation))
s = s.lower()

doc = nlp(s) # doc is a list of tokens, e.g. doc[0] is 'horrible'
doc.ents = tuple() # so far delete any pre-predicted entities

matches = matcher(doc) # matches is a list of tuples, e.g. [(4851363122962674176, [6, 0, 10, 9])]
                       # one tuple is match_id and tokens indices

print(f'\nNumber of matches: {len(matches)}')

displacy.serve(doc, style='ent', page=True, options={'ents': ['DIS']})
