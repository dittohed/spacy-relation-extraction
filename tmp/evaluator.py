import glob
import re

import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher, DependencyMatcher
from spacy import displacy

import extract_diseases as ed

# true labels should be denoted with enclosing semi-colons exactly this way
# ;;<token>;;
# or (in case of N tokens entity)
# ;;<token1> ... <tokenN>;;
TL_EXPRESSION = ';;.*?;;' # non-greedy regexp for identifying true labels

DIR_PATH = './articles/diseases'
generate_html = True # whether to save results to HTML using displacy for each article

nlp = spacy.load('en_core_web_trf')

tp = 0 # true positive
fp = 0 # false positive
fn = 0 # false negative

articles = glob.glob(f'{DIR_PATH}/*_test.txt')
for article in articles:
    with open(article, 'r') as file:
        matcher = Matcher(nlp.vocab)
        matcher_dep = DependencyMatcher(nlp.vocab)

        article_id = article.split('/')[-1].split('.')[0]
        print(f'Handling article {article_id}...')

        lines = file.readlines()
        text = ' '.join(lines)

        # --- reading true labels ---
        tl_found = 0
        tl_positions = []
        for match in re.finditer(TL_EXPRESSION, text): # matches are returned in left-to-right order
            start, end = match.span()
            # print(f'Matched text: {text[start : end]}')

            tl_positions.append((start-(tl_found)*4, end-(tl_found+1)*4))
            tl_found += 1
        # print(f'Found {tl_found} true labels (entities)...')
        #
        # print(tl_positions)
        # print(text.replace(';;', ''))
        doc = nlp(text.replace(';;', ''))

        tl_spans = tuple([doc.char_span(start, end, label='DIS') for (start, end) in tl_positions])

        # --- extracting labels ---
        doc.ents += ed.match_initialisms(doc)

        matcher_dep.add('dependencies', ed.dependencies_patterns,
                    on_match=ed.add_disease_ent_dep)
        matcher_dep(doc) # matches is a list of tuples, e.g. [(4851363122962674176, [6, 0, 10, 9])]
                         # one tuple is match_id and tokens indices

        matcher.add('standalones', ed.standalones_patterns,
                    on_match=ed.add_disease_ent)
        matcher(doc) # matches is a list of tuples, e.g. [(4851363122962674176, [23, 24)]
                     # one tuple is match_id, match start and match end

        # get extracted labels spans in order to compare it with true label spans
        el_spans = doc.ents

        # --- comparing results ---
        tl_spans = set([(span.start, span.end) for span in tl_spans])
        el_spans = set([(span.start, span.end) for span in el_spans if span.label_ == 'DIS'])

        doc.ents = tuple() # reset entities
        for tl_span in tl_spans:
            if tl_span in el_spans:
                entity = Span(doc, tl_span[0], tl_span[1], label='TP')
                tp +=1
            else:
                entity = Span(doc, tl_span[0], tl_span[1], label='FN')
                fn += 1

            doc.ents += (entity,)

        for el_span in el_spans:
            entity = Span(doc, el_span[0], el_span[1], label='FP')
            try:
                doc.ents += (entity,)
            except ValueError:
                pass # that wasn't a FP
            else:
                fp += 1

        if generate_html:
            html = displacy.render(doc, style='ent', page=True,
                                 options={'colors': {'TP': '#00FF00', 'FN': '#FF0000', 'FP': '#FF00FF'}})
            with open(f'./displacy/{article_id}.html', 'w') as html_file:
                print(f'Saving a generated file to {article_id}.html')
                html_file.write(html)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(f'Precision = {precision}, Recall = {recall}')
