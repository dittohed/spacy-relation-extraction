import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Span
from spacy import displacy

import re
import string

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
    'symptoms'
]

# regexp for identifying "base" words to build on
# for example this might be sole virus or Coronavirus
keywords_regexp = '^.+(is|us|ism|ysm|virus|pathy|pox|ia|cocci|ae)$'

# words to exclude from regexp above
ex_regexp = (
    'this',
    'various',
    'prognosis',
    'hypothesis'
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
    'consensus')

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

# modifiers modify "base" words
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

# patterns for rule-based matching
pattern1 = [
    # for example matches "heavy depression"
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': keywords}}
    },
    modifier
]

pattern2 = [
    # for example matches "bloom- associated illness"
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': keywords}}
    },
    modifier,
    modmodifier
]

pattern3 = [
    # for example matches "Aortic Aneurysm"
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'REGEX': keywords_regexp, 'NOT_IN': ex_regexp}}
    },
    modifier
]

pattern4 = [
    # for example matches "Acute Flaccid Myelitis"
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'REGEX': keywords_regexp, 'NOT_IN': ex_regexp}}
    },
    modifier,
    modmodifier
]

# order does matter
dependencies_patterns = [
    pattern2,
    pattern4,
    pattern1,
    pattern3
]

# "base" words with no modifiers
standalones_patterns = [
    # for example matches "diarrhea"
    [{'LEMMA': {'REGEX': keywords_regexp, 'NOT_IN': ex_regexp}}],
    [{'LEMMA': {'IN': keywords}}]
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

def add_disease_ent(matcher, doc, i, matches):
    """
    Creates entity label for current match resulting from matching standalones.
    """

    global initialisms_ents

    match_id, start, end = matches[i]
    entity = Span(doc, start, end, label='DIS')

    try:
        doc.ents += (entity,)
    except ValueError:
        pass

def add_disease_ent_dep(matcher, doc, i, matches):
    """
    Creates entity label for current match resulting from dependecy tree.
    """

    match_id, token_ids = matches[i]
    start = min(token_ids)
    end = max(token_ids) + 1

    entity = Span(doc, start, end, label='DIS')

    try:
        doc.ents += (entity,)
    except ValueError:
        pass # Span simply won't be added
