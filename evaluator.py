# cel:
# puścić regexpa, żeby znalazł wszystkie true labele
# usunięcie znaczników + puścić nasze narzędzie NER
# porównać spany
# wyświetlić statystyki oraz wizualizację

# potrzeba:
# oznaczyć jakoś prawdziwe choroby w tekście (możliwe wiele tokenów na jedną nazwę)
# zapisać Spany (tekst, początek i koniec, ale tak, żeby dało się to porównać z tekstem bez oznaczeń)
# - możliwe, że tutaj nie muszą w ogóle tworzyć żadnego doca
# przepuścić tekst przez nasze narzędzie NER
# porównać Spany

import glob
import re

import spacy
from spacy.tokens import Span
from spacy import displacy

DIR_PATH = './articles/diseases'
GENERATE = True # whether to save results to HTML using displacy for each article

# token of interest should be denoted with enclosing semi-colons exactly this way ;;<token>;;
toi_expression = ';;.*?;;' # non-greedy

nlp = spacy.load('en_core_web_trf')

articles = glob.glob(f'{DIR_PATH}/*_test.txt')

for article in articles:
    with open(article, 'r') as file:
        article_id = article.split('/')[-1].split('.')[0]
        print(f'Handling article {article_id}...')

        lines = file.readlines()
        text = ' '.join(lines)
        text = 'asdasd, ;;aa sasa;;, ;;b;; ...'

        toi_found = 0
        toi_positions = []
        for match in re.finditer(toi_expression, text): # matches are returned in left-to-right order
            start, end = match.span()
            print(f'Matched text: {text[start : end]}')

            toi_positions.append((start-(toi_found+1)*2, end-(toi_found+1)*3))
            toi_found += 1

        print(toi_positions)
        print(text.replace(';;', ''))

        doc = nlp(text.replace(';;', ''))

        toi_ents = (doc.char_span(start, end, label='DIS_TRUE') for (start, end) in toi_positions)
        doc.ents = toi_ents

        print('Ents:')
        print([(ent.text, ent.start, ent.end) for ent in list(doc.ents)])

        for tok in doc:
            print(tok)

        if GENERATE:
            html = displacy.render(doc, style='ent', page=True, options={'ents': ['DIS_TRUE']})
            with open(f'./displacy/{article_id}.html', 'w') as html_file:
                html_file.write(html)
