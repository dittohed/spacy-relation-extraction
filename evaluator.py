import glob
import re

import spacy
from spacy.tokens import Span
from spacy import displacy

DIR_PATH = './articles/diseases'
GENERATE = True # whether to save results to HTML using displacy for each article

# token of interest should be denoted with enclosing semi-colons exactly this way ;;<token>;;
toi_expression = ';;.*;;'

nlp = spacy.load('en_core_web_trf')

articles = glob.glob(f'{DIR_PATH}/*_test.txt')

for article in articles:
    with open(article, 'r') as file:
        article_id = article.split('/')[-1].split('.')[0]
        print(f'Handling article {article_id}...')

        lines = file.readlines()
        text = ' '.join(lines)

        doc = nlp('asdasd, ;;aa sasa;;, ...')

        # for sent in doc.sents:
        for match in re.finditer(toi_expression, doc.text):
            start, end = match.span()

            entity = doc.char_span(start, end, label='DIS_TRUE')
            print(f'Matched text: {entity.text}')

            try:
                doc.ents += (entity,)
            except ValueError:
                pass

        if GENERATE:
            html = displacy.render(doc, style='ent', page=True, options={'ents': ['DIS_TRUE']})
            with open(f'./displacy/{article_id}.html', 'w') as html_file:
                html_file.write(html)
