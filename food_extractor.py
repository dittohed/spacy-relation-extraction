import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Span
from spacy import displacy

import diseases_extractor as dis

import string

# TODO: fix no MDS

anchors = ['consumption', 'intake', 'serving', 'consume', 'eat', 'portion', 'cup']
keywords = ['food', 'diet', 'vegetable', 'fruit', 'meal']
initialisms = ['MDS', 'DASH', 'NFI']

def add_modifier(left_id):
    """
    Returns modifier for a specified node.
    """

    modifier = {
        'LEFT_ID': left_id,
        'REL_OP': '>>',
        'RIGHT_ID': 'modifier',
        'RIGHT_ATTRS': {'DEP': {'IN': ['amod', 'compound', 'poss', 'nmod', 'npadvmod']}}
    }

    return modifier

def add_conj(left_id):
    """
    Returns conjunction for a specified node.
    """

    conj = {
        'LEFT_ID': left_id,
        'REL_OP': '>>',
        'RIGHT_ID': 'conj',
        'RIGHT_ATTRS': {'DEP': 'conj'}
    }

    return conj

def add_appos(left_id):
    """
    Returns appos for a specified node.
    """

    appos = {
    'LEFT_ID': left_id,
    'REL_OP': '>>',
    'RIGHT_ID': 'appos',
    'RIGHT_ATTRS': {'DEP': 'appos'}
    }

    return appos

pattern_base1 = [
    # matches consumption/intake/portion/cup/serving of [FOOD]
    # with possible modifications (using functions defined above)
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': ['consumption', 'intake', 'portion', 'cup', 'serving']}}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>',
        'RIGHT_ID': 'preposition',
        'RIGHT_ATTRS': {'LOWER': 'of'}
    },
    {
        'LEFT_ID': 'preposition',
        'REL_OP': '>',
        'RIGHT_ID': 'object of preposition',
        'RIGHT_ATTRS': {'DEP': 'pobj'}
    }
]

pattern_base2 = [
    # matches [FOOD] intake/consumption
    # with possible modifications (using functions defined above)
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': ['intake', 'consumption']}}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>',
        'RIGHT_ID': 'food modifier',
        'RIGHT_ATTRS': {'DEP': {'IN': ['compound', 'poss', 'nmod', 'npadvmod']}}
    }
]

pattern_base3 = [
    # such pattern matches eat/consume [FOOD]
    # with possible modifications (using functions defined above)
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': ['consume', 'consuming', 'eat', 'eating']}}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>',
        'RIGHT_ID': 'dobj',
        'RIGHT_ATTRS': {'DEP': 'dobj'}
    }
]

pattern_base4 = [
    # such patterns matches food / diet / vegetable / fruit / meal
    # with possible modifications (using functions defined above)
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': keywords}}
    }
]

pattern_base5 = [
    # such patterns matches MDS / DASH / NFI
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'ORTH': {'IN': initialisms}}
    }
]

dependencies_patterns = [
    pattern_base1+[add_modifier('object of preposition')],
    pattern_base1,
    pattern_base1+[add_conj('object of preposition'), add_modifier('conj')],
    pattern_base1+[add_conj('object of preposition')],
    pattern_base1+[add_appos('object of preposition'), add_modifier('appos')],
    pattern_base1+[add_appos('object of preposition')],

    pattern_base2+[add_modifier('food modifier')],
    pattern_base2,
    pattern_base2+[add_conj('food modifier'), add_modifier('conj')],
    pattern_base2+[add_conj('food modifier')],
    pattern_base2+[add_appos('food modifier'), add_modifier('appos')],
    pattern_base2+[add_appos('food modifier')],

    pattern_base3+[add_modifier('dobj')],
    pattern_base3,
    pattern_base3+[add_conj('dobj'), add_modifier('conj')],
    pattern_base3+[add_conj('dobj')],
    pattern_base3+[add_appos('dobj'), add_modifier('appos')],
    pattern_base3+[add_appos('dobj')],

    pattern_base4+[dis.add_modifier('anchor', 'modifier')],
    pattern_base4,

    pattern_base5
]

def add_food_dep(matcher, doc, i, matches):
    """
    Creates entity label for current match resulting from dependecy tree.
    """

    match_id, token_ids = matches[i]

    start = min(token_ids)
    end = max(token_ids) + 1

    for token_id in token_ids:
        if doc[token_id].pos_ in ('NOUN', 'ADJ', 'VERB') \
        and doc[token_id].lemma_ not in anchors \
        and doc[token_id].tag_ not in ['JJR']:
            entity = Span(doc, token_id, token_id+1, label='FOOD')

            try:
                doc.ents += (entity,)
            except ValueError:
                pass # Span simply won't be added

def merge_entities(doc):
    """
    Merges FOOD entities when they occur next to each other.
    """

    entities = tuple()
    spans_start = [span.start for span in doc.ents if span.label_ == 'FOOD']
    behind_to_merge = 0

    for i, id in enumerate(spans_start):
        if i + 1 < len(spans_start) and spans_start[i + 1] == id + 1:
            behind_to_merge += 1
        else:
            entities += (Span(doc, id - behind_to_merge, id + 1, label='FOOD'),)
            behind_to_merge = 0

    doc.ents = tuple([span for span in doc.ents if span.label_ != 'FOOD']) # removing food entities
    doc.ents += entities
