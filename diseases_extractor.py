import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Span
from spacy import displacy

import re
import string

# TODO: analysis, zahardcode'ować initiliasmy i dodać regexp z bacter
# Unstable angina
# Esophageal cancer

# list of "base" words to build on
# for example this might be sole "fever" or "yellow fever"
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
    'asthma',
    'symptoms']

# regexp for identifying "base" words to build on
# for example this might be sole virus or Coronavirus
keywords_regexp = '^.+(is|us|ism|ysm|virus|pathy|pox|ia|cocci|ae)$'

# words to exclude from regexp above
ex_regexp = [
    'this',
    'various',
    'prognosis',
    'hypothesis',
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
    'criteria',
    'consensus']

# most commonly occuring initiliasms (in medical articles) to exclude
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
    'DNA',
    'MDS',
    'DASH',
    'NFI')

bacter_regexp = '^.+bacter'

def add_modifier(left_id, right_id):
    """
    Returns modifier for a specified node.
    """

    modifier = {
        'LEFT_ID': left_id,
        'REL_OP': '>',
        'RIGHT_ID': right_id,
        'RIGHT_ATTRS': {'DEP': {'IN': ['amod', 'compound', 'poss', 'nmod', 'npadvmod']}}
    }

    return modifier

# patterns for rule-based matching
pattern_base1 = [
    # for example matches "depression" or "heavy depression" (with modifier appended below)
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': keywords}}
    }
]

pattern_base2 = [
    # for example matches "aunerysm" or "Aortic Aneurysm" (with modifier appended below)
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'REGEX': keywords_regexp, 'NOT_IN': ex_regexp}}
    }
]

# order does matter
dependencies_patterns = [
    pattern_base1+[add_modifier('anchor', 'modifier'), add_modifier('modifier', 'modmodifier')],
    pattern_base2+[add_modifier('anchor', 'modifier'), add_modifier('modifier', 'modmodifier')],
    pattern_base1+[add_modifier('anchor', 'modifier')],
    pattern_base2+[add_modifier('anchor', 'modifier')],
    pattern_base1,
    pattern_base2
]

def match_initialisms(doc):
    """
    Using a separate function on account tokenizing issues with -.
    """

    initialisms_regexp = r'\b[A-Z0-9]+-?[A-Z0-9]+\b'
    num_regexp = r'\b[0-9]+-?[0-9]+\b'

    initialisms_ents = tuple()
    for match in re.finditer(initialisms_regexp, doc.text):
        start, end = match.span()
        if re.compile(num_regexp).search(doc.text[start : end]) or doc.text[start : end] in ex_initialisms:
            continue # e.g. 4343 or 323-1233 or API was found

        entity = doc.char_span(start, end, label='DIS', alignment_mode='expand')

        try:
            doc.ents += (entity,)
        except Exception as e:
            pass # actually, it's probably an organization
        else:
            initialisms_ents += (entity,)

    return initialisms_ents

def add_disease_ent_dep(matcher, doc, i, matches):
    """
    Creates entity label for current match resulting from dependecy tree.
    """

    match_id, token_ids = matches[i]
    start = min(token_ids)
    end = max(token_ids) + 1

    entity = Span(doc, start, end, label='DIS')
    # print(f'Matched span: {entity.text}')

    try:
        doc.ents += (entity,)
    except ValueError:
        pass # Span simply won't be added
