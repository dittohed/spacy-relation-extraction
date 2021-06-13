import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

patterns = [
    [{'POS': 'ADJ'}, {'LOWER': 'infection'}],
    [{}, {'LOWER': 'syndrome'}],
    [{}, {'TEXT': {'REGEX': '[A-Za-z]*(is|us|ism|ysm)'}}]
]

def add_disease_ent(matcher, doc, i, matches):
    # Get the current match and create tuple of entity label, start and end.
    # Append entity to the doc's entity. (Don't overwrite doc.ents!)
    match_id, start, end = matches[i]
    entity = Span(doc, start, end, label='DIS')
    print(entity.text)
    doc.ents += (entity,)

matcher.add("disease_pattern", patterns, on_match=add_disease_ent)
doc = nlp("Blablabla Aortic Aneurysm blalba Adenovirus Infection blabla Acquired Immune Deficiency Syndrome")
matches = matcher(doc)

displacy.serve(doc, style='ent', page=True, options={'ents': ['DIS']})
