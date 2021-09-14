import spacy
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Span
from spacy import displacy
from spacy.util import filter_spans

association_keywords = [
    'association',
    'relation',
    'connection',
    'relationship',
    'link'
]

association_verb_keywords = [
    'associate',
    'relate',
    'connect',
    'link',
    'cause',
    'increase',
    'effect',
    'affect'
]

pattern1 = [
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': association_keywords}}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>>',
        'RIGHT_ID': 'food',
        'RIGHT_ATTRS': {'ENT_TYPE': 'FOOD'}
    },
    {
        'LEFT_ID': 'food',
        'REL_OP': ';*',
        'RIGHT_ID': 'dis',
        'RIGHT_ATTRS': {'ENT_TYPE': 'DIS'}
    },
]

pattern1_inverse = [
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': association_keywords}}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>>',
        'RIGHT_ID': 'dis',
        'RIGHT_ATTRS': {'ENT_TYPE': 'DIS'}
    },
    {
        'LEFT_ID': 'dis',
        'REL_OP': ';*',
        'RIGHT_ID': 'food',
        'RIGHT_ATTRS': {'ENT_TYPE': 'FOOD'}
    },
]

pattern2 = [
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': association_verb_keywords}}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>>',
        'RIGHT_ID': 'food',
        'RIGHT_ATTRS': {'DEP': {'IN': ['nsubjpass', 'compound']}, 'ENT_TYPE': 'FOOD'},
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>>',
        'RIGHT_ID': 'dis',
        'RIGHT_ATTRS': {'ENT_TYPE': 'DIS'}
    }
]

pattern2_inverse = [
    {
        'RIGHT_ID': 'anchor',
        'RIGHT_ATTRS': {'LEMMA': {'IN': association_verb_keywords}}
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>>',
        'RIGHT_ID': 'dis',
        'RIGHT_ATTRS': {'ENT_TYPE': 'DIS'},
    },
    {
        'LEFT_ID': 'anchor',
        'REL_OP': '>>',
        'RIGHT_ID': 'food',
        'RIGHT_ATTRS': {'ENT_TYPE': 'FOOD'}
    }
]

relations_patterns = [
    pattern1,
    pattern1_inverse,
    pattern2,
    pattern2_inverse
]


def add_relations_ent_dep(matcher, doc, i, matches):
    '''
    Creates entity label for current match resulting from dependecy tree.
    '''

    match_id, token_ids = matches[i]
    start = min(token_ids)
    end = max(token_ids) + 1

    entity = Span(doc, start, end, label='REL')

    # consider relations only within current sentence / line
    if '. ' in entity.text or '\n' in entity.text:
        return

    spans = ([*doc.ents, entity])

    try:
        # chose longer spans when overlap occurs as relations will be longer than food / disease
        doc.ents = filter_spans(spans)
        #doc.ents += (entity,)
    except Exception as e:
        pass