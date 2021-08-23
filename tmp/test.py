# displays dependency tree for a given sentence
import spacy
from spacy import displacy

# TODO: ładować plik linijka po linijce i tylko zmieniać indeks

nlp = spacy.load('en_core_web_trf')
doc = nlp('study investigating interactions between genetic and lifestyle factors')
# print(doc.ents)
# for word in doc:
#     print(word.text, word.lemma_)
displacy.serve(doc, style='dep')
# print([token.dep_ for token in doc])
