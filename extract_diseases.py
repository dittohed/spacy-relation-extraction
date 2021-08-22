# TODO:
# - exclude keywords_regexp most common words (longer than 4), add to stop_words.txt
# - separate modules/files (config file)
# - optimization; for now, the text is scanned 7 times (each pass for each pattern)?

import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Span
from spacy import displacy

import re
import string

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

keywords_regexp = '^.+(is|us|ism|ysm|virus|pathy|pox|ia)$'

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

dependencies_patterns = [
    pattern2,
    pattern4,
    pattern1,
    pattern3
]

# --- standalones ---
standalones_patterns = [
    [{'LOWER': {'REGEX': keywords_regexp}}],
    [{'LOWER': {'IN': ['flu', 'diarrhea', 'cold']}}]
]

def match_initialisms(doc):
    '''
    Using a separate function on account tokenizing issues with -.
    '''

    initialisms_regexp = r'\b[A-Z0-9]+-?[A-Z0-9]+\b'
    num_regexp = r'\b[0-9]+-?[0-9]+\b'

    initialisms_ents = tuple()
    for match in re.finditer(initialisms_regexp, doc.text):
        start, end = match.span()
        if re.compile(num_regexp).search(doc.text[start : end]):
            continue # e.g. 4343 or 323-1233 was found

        entity = doc.char_span(start, end, label='DIS')
        print(f'Matched text: {entity.text}')

        try:
            doc.ents += (entity,)
        except ValueError: # actually, it's probably an organization
            print('Nope.')
        else:
            initialisms_ents += (entity,)

    return initialisms_ents

def add_disease_ent(matcher, doc, i, matches):
    '''
    Creates entity label for current match resulting from matching standalones.
    '''

    global initialisms_ents

    match_id, start, end = matches[i]
    entity = Span(doc, start, end, label='DIS')
    print(f'Matched text: {entity.text}')

    try:
        doc.ents += (entity,)
    except ValueError:
        pass

def add_disease_ent_dep(matcher, doc, i, matches):
    '''
    Creates entity label for current match resulting from dependecy tree.
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

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_trf')
    matcher = Matcher(nlp.vocab)
    matcher_dep = DependencyMatcher(nlp.vocab)

    # read file
    with open('./articles/diseases/PMC5363789_true.txt', 'r') as file:
        # lines = file.readlines()
        # text = ' '.join(lines)

        s = '''
        Furthermore, two other chronic diseases, EV-D68, USA, COVID-19, liver cirrhosis and interstitial lung disease/lungfibrosis, were also associated with a poor prognosis.
        '''

        doc = nlp(s) # doc is a list of tokens, e.g. doc[0] is 'horrible'

        doc.ents += match_initialisms(doc)

        # patterns order in patterns list does matter
        matcher_dep.add('dependencies', dependencies_patterns,
                    on_match=add_disease_ent_dep)
        matcher_dep(doc) # matches is a list of tuples, e.g. [(4851363122962674176, [6, 0, 10, 9])]
                         # one tuple is match_id and tokens indices

        matcher.add('standalones', standalones_patterns,
                    on_match=add_disease_ent)
        matcher(doc) # matches is a list of tuples, e.g. [(4851363122962674176, [23, 24)]
                     # one tuple is match_id, match start and match end

        for tok in doc:
            print(tok)

        displacy.serve(doc, style='ent', page=True, options={'ents': ['DIS']})
