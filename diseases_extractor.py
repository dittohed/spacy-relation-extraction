# TODO:
# - optimization; for now, the text is scanned 7 times (each pass for each pattern)?
# - remove if main

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
    'ilness',
    'influenza',
    'cholera',
    'diabetes',
    'depression',
    'neoplasm',
    'asthma'
]

keywords_regexp = '^.+(is|us|ism|ysm|virus|pathy|pox|ia|cocci|ae)$'
ex_regexp = (
    'this',
    'various',
    'prognosis',
    'analysis',
    'diagnosis',
    'status',
    'previous',
    'ambiguous',
    'his',
    'focus',
    'mechanism',
    'organism',
    'microorganism',
    'thus',
    'emphasis',
    'homeostasis',
    'via',
    'continuous',
    'analogous',
    'cholera')

ex_initialisms = (
    'AND',
    'OR',
    'OWL',
    'API',
    'SPARQL',
    'RDF',
    '3D',
    'MP4',
    'USA',
    'DNA')

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
        'RIGHT_ATTRS': {'LEMMA': {'IN': keywords}}
    },
    # modifier specification
    modifier
]

pattern2 = [
    # anchor specification
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': keywords}}
    },
    modifier, # modifier specification
    modmodifier # modifier's modifiers specification
]

pattern3 = [
    # anchor specification
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'REGEX': keywords_regexp, 'NOT_IN': ex_regexp}}
    },
    modifier
]

pattern4 = [
    # anchor specification
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'REGEX': keywords_regexp, 'NOT_IN': ex_regexp}}
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
    [{'LEMMA': {'REGEX': keywords_regexp, 'NOT_IN': ex_regexp}}],
    [{'LEMMA': {'IN': ['flu', 'diarrhea', 'cold']}}]
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
        if re.compile(num_regexp).search(doc.text[start : end]) or doc.text[start : end] in ex_initialisms:
            continue # e.g. 4343 or 323-1233 was found

        entity = doc.char_span(start, end, label='DIS', alignment_mode='expand')
        # print(f'Matched text: {entity.text}')

        try:
            doc.ents += (entity,)
        except ValueError:
            # print('Nope.')
            pass # actually, it's probably an organization
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
    # print(f'Matched text: {entity.text}')

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
    # print(f'Matched text: {entity.text}')

    try:
        doc.ents += (entity,)
    except ValueError:
        pass # Span simply won't be added

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_trf')
    matcher = Matcher(nlp.vocab)
    matcher_dep = DependencyMatcher(nlp.vocab)

    # read file
    with open('./articles/diseases/PMC5363789_curr.txt', 'r') as file:
        lines = file.readlines()
        text = ' '.join(lines)

        s = '''
        This DIS is the first report to perform follow-up survival analysis DIS across various DIS common diseases.
        Among the 32 diseases, dyslipidemia DIS was the most common disease DIS (n = 37,478), while gallbladder/cholangiocarcinoma were the least common (n = 366) (Table 1).
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
            print(tok, tok.pos_, tok.tag_)

        # displacy.serve(doc, style='ent', page=True, options={'ents': ['DIS']})
        displacy.serve(doc, style='ent', page=True)
