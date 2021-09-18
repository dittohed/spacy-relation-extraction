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
    'consensus',
    'campus',
    'versus']

bacter_regexp = '^.+bacter'

# reading initialisms.txt into a list
with open('./initialisms.txt', 'r') as file:
    lines = file.readlines()
    initialisms = [line.strip() for line in lines]

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

pattern_base3 = [
    # for example matches
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'REGEX': bacter_regexp}}
    }
]

pattern_base4 = [
    # for example matches "ADHD"
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'ORTH': {'IN': initialisms}}
    }
]

# order in the list does matter
dependencies_patterns = [
    pattern_base1+[add_modifier('anchor', 'modifier'), add_modifier('modifier', 'modmodifier')],
    pattern_base2+[add_modifier('anchor', 'modifier'), add_modifier('modifier', 'modmodifier')],
    pattern_base1+[add_modifier('anchor', 'modifier')],
    pattern_base2+[add_modifier('anchor', 'modifier')],
    pattern_base1,
    pattern_base2,
    pattern_base3,
    pattern_base4
]

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
