# TODO: klastry (może po prostu przejście po entity pod koniec i sklejenie)
# + różne możliwości (raczej nie ma optionalów)

import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Span
from spacy import displacy

import string

anchors = ['consumption', 'intake', 'serving', 'consume']

def add_modifier(left_id):
    """
    Adds modifier to a specified node.
    """

    modifier = {
        'LEFT_ID': left_id,
        'REL_OP': '>',
        'RIGHT_ID': 'modifier',
        'RIGHT_ATTRS': {'DEP': {'IN': ['amod', 'compound', 'poss', 'nmod', 'npadvmod']}}
    }

    return modifier

def add_conj(left_id):
    """
    Adds conjunction to a specified node.
    """

    conj = {
        'LEFT_ID': left_id,
        'REL_OP': '>>',
        'RIGHT_ID': 'conj', # FIX
        'RIGHT_ATTRS': {'DEP': 'conj'}
    }

    return conj

def add_appos(left_id):
    """
    Adds appos to a specified node.
    """

    appos = {
    'LEFT_ID': left_id,
    'REL_OP': '>>',
    'RIGHT_ID': 'appos', # FIX
    'RIGHT_ATTRS': {'DEP': 'appos'}
    }

    return appos

pattern_base1 = [
    # such pattern matches consumption of [FOOD]
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LOWER': 'consumption'}
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
    # such pattern matches [FOOD] consumption
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LOWER': 'consumption'}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>',
        'RIGHT_ID': 'food modifier',
        'RIGHT_ATTRS': {'POS': 'NOUN'}
    }
]

pattern_base3 = [
    # such pattern matches consume [FOOD]
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': 'consume'}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>',
        'RIGHT_ID': 'dobj',
        'RIGHT_ATTRS': {'DEP': 'dobj'}
    }
]

# FIX: merge 1 and 4
pattern_base4 = [
    # such pattern matches intake of [FOOD]
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': 'intake'}
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

pattern_base5 = [
    # such pattern matches [FOOD] intake
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': 'intake'}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>',
        'RIGHT_ID': 'food modifier',
        'RIGHT_ATTRS': {'DEP': {'IN': ['amod', 'compound', 'poss', 'nmod', 'npadvmod']}}
    }
]

pattern_base6 = [
    # such pattern matches servings of [FOOD]
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': 'serving'}
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

dependencies_patterns = [
    pattern_base1+[add_modifier('object of preposition')],
    pattern_base1,
    pattern_base1+[add_conj('object of preposition'), add_modifier('conj')],
    pattern_base1+[add_conj('object of preposition')],
    pattern_base2+[add_modifier('food modifier')],
    pattern_base2,
    pattern_base2+[add_conj('food modifier'), add_modifier('conj')],
    pattern_base2+[add_conj('food modifier')],
    pattern_base3+[add_modifier('dobj')],
    pattern_base3,
    pattern_base3+[add_conj('dobj'), add_modifier('conj')],
    pattern_base3+[add_conj('dobj')],
    pattern_base4+[add_modifier('object of preposition')],
    pattern_base4,
    pattern_base4+[add_conj('object of preposition'), add_modifier('conj')],
    pattern_base4+[add_conj('object of preposition')],
    pattern_base4+[add_appos('object of preposition'), add_modifier('appos')],
    pattern_base4+[add_appos('object of preposition')],
    pattern_base5+[add_modifier('food modifier')],
    pattern_base5,
    pattern_base5+[add_conj('food modifier'), add_modifier('conj')],
    pattern_base5+[add_conj('food modifier')],
    pattern_base6+[add_modifier('object of preposition')],
    pattern_base6,
    pattern_base6+[add_conj('object of preposition'), add_modifier('conj')],
    pattern_base6+[add_conj('object of preposition')],
]

def add_disease_food_dep(matcher, doc, i, matches):
    """
    Creates entity label for current match resulting from dependecy tree.
    """

    match_id, token_ids = matches[i]
    print(f'Matched tokens: {[(doc[token_id].text, doc[token_id].pos_) for token_id in token_ids]}')

    start = min(token_ids)
    end = max(token_ids) + 1
    print(f'Matched span: {doc[start : end]}')

    # entity = Span(doc, start, end, label='FOOD')
    # dla złożonych jedzeń:
    # znajdź modyfikator, znajdź rodzica, zrób Span

    for token_id in token_ids:
        if doc[token_id].pos_ in ('NOUN', 'ADJ', 'VERB') and doc[token_id].lemma_ not in anchors:
            # użyć tag zamiast pos_ i wykluczyć JJ, JJR itd.
            entity = Span(doc, token_id, token_id+1, label='FOOD')
            print(f'Matched text: {entity.text}')

            try:
                doc.ents += (entity,)
            except ValueError:
                print(f'Nope...') # Span simply won't be added

def merge_entities(doc):
    """
    Merges FOOD entities when they occur next to each other.
    """

    entities = tuple()
    spans_start = [span.start for span in doc.ents]
    behind_to_merge = 0

    for i, id in enumerate(spans_start):
        if i + 1 < len(spans_start) and spans_start[i + 1] == id + 1:
            behind_to_merge += 1
        else:
            print(f'Merging {doc[id - behind_to_merge : id + 1]}')
            entities += (Span(doc, id - behind_to_merge, id + 1, label='FOOD'),)
            behind_to_merge = 0

    doc.ents = entities

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_trf')
    matcher = Matcher(nlp.vocab)
    matcher_dep = DependencyMatcher(nlp.vocab)

    doc = nlp('Diet data were processed to derive a Mediterranean diet score (MDS) and daily servings of fruit and vegetables, and fish.')
    doc.ents = tuple()

    # patterns order in patterns list does matter
    matcher_dep.add('dependencies', dependencies_patterns,
                on_match=add_disease_food_dep)
    matcher_dep(doc) # matches is a list of tuples, e.g. [(4851363122962674176, [6, 0, 10, 9])]
                     # one tuple is match_id and tokens indices

    for tok in doc:
        print(tok.text, tok.pos_, tok.dep_, tok.tag_, tok.lemma_)

    merge_entities(doc)

    # displacy.serve(doc, style='dep')
    displacy.serve(doc, style='ent', page=True, options={'ents': ['FOOD']})
