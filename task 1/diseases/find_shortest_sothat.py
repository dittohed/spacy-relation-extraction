# helper script for filtering the shortest disease name ending with X
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_trf')
matcher = Matcher(nlp.vocab)

pattern = [{'TEXT': {'REGEX': '^.+(is|us|ism|ysm|virus|pathy|pox|ia)$'}}]
matcher.add('pattern', [pattern])

with open('Diseases & Conditions A-Z.txt', 'r') as f:
    s = f.read().replace('\n', ' ')

doc = nlp(s)
matches = matcher(doc)

min_len = 10
for _, start, end in matches:
    if len(doc[start : end].text) <= min_len:
        min_len = len(doc[start : end].text)
        print(f'{doc[start : end]} ({min_len})')
