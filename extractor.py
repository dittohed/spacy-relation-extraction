# evaluates or predicts using specified extractor
# example of use: python extractor.py evaluate diseases ./articles/diseases --no-html 1

# TODO: przetestować food_extractor, połączyć food z disease + final eval

import argparse
import glob
import re

import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher, DependencyMatcher
from spacy import displacy

import diseases_extractor as dis
import food_extractor as food
# import relations_extractor as rel

class Extractor:
    # true labels should be denoted with enclosing semi-colons exactly this way
    # ;;<token>;;
    # or (in case of N tokens entity)
    # ;;<token1> ... <tokenN>;;
    TL_EXPRESSION = ';;.*?;;' # non-greedy regexp for identifying true labels

    def __init__(self, task, domain, datapath, nohtml, nolabeling):
        self.task = task
        self.domain = domain
        self.datapath = datapath
        self.nohtml = nohtml
        self.nolabeling = nolabeling

        self.nlp = spacy.load('en_core_web_trf')

    def run(self):
        if self.task == 'evaluate':
            self.evaluate()
        else:
            self.predict()

    def extract_labels(self, doc):
        """
        Extract labels using a proper extractor.
        """

        matcher = Matcher(self.nlp.vocab)
        matcher_dep = DependencyMatcher(self.nlp.vocab)

        if self.domain == 'diseases':
            doc.ents += dis.match_initialisms(doc)

            matcher_dep.add('dependencies', dis.dependencies_patterns,
                        on_match=dis.add_disease_ent_dep)
            matcher_dep(doc)

            matcher.add('standalones', dis.standalones_patterns,
                        on_match=dis.add_disease_ent)
            matcher(doc)

        elif self.domain == 'food':
            doc.ents = tuple([ent for ent in doc.ents if ent.label_ in ('PERSON', 'ORG', 'GPE', 'DIS')])

            matcher_dep.add('dependencies', food.dependencies_patterns,
                        on_match=food.add_food_dep)
            matcher_dep(doc)

            food.merge_entities(doc)

        elif self.domain == 'both':
            pass
        else:
            pass

    def predict(self):
        """
        Predicts entities and optionally labels them in a new *_pred.txt file.
        """

        articles = glob.glob(f'{self.datapath}/*_true.txt')

        for article in articles:
            with open(article, 'r') as file:
                article_id = article.split('/')[-1].split('.')[0]
                print(f'Handling article {article_id}...')

                lines = file.readlines()
                text = ' '.join(lines)

                doc = self.nlp(text)
                self.extract_labels(doc)

                if not self.nohtml:
                    self.generate_html(doc, f'./displacy/{article_id}_pred.html')

                if not self.nolabeling:
                    pass
                    # TODO: add :: to read file

    def evaluate(self):
        """
        Evaluates extraction method by comparing labeled (TL_EXPRESSION) .txt files
        and extracted labels.
        Prints out precision and recall.
        """

        tp = 0 # true positive
        fp = 0 # false positive
        fn = 0 # false negative

        articles = glob.glob(f'{self.datapath}/*_test.txt')

        for article in articles:
            with open(article, 'r') as file:
                article_id = article.split('/')[-1].split('.')[0]
                print(f'Handling article {article_id}...')

                lines = file.readlines()
                text = ' '.join(lines)

                # reading true labels and creating Span objects
                tl_found = 0
                tl_positions = []
                for match in re.finditer(self.TL_EXPRESSION, text): # matches are returned in left-to-right order
                    start, end = match.span()

                    tl_positions.append((start-(tl_found)*4, end-(tl_found+1)*4))
                    tl_found += 1

                doc = self.nlp(text.replace(';;', '')) # labels not necessary anymore
                tl_spans = tuple([doc.char_span(start, end, label='DIS') for (start, end) in tl_positions])

                self.extract_labels(doc)

                el_spans = doc.ents # extracted labels (Span objects)

                # comparing true labels with extracted labels
                tl_spans = set([(span.start, span.end) for span in tl_spans])
                el_spans = set([(span.start, span.end) for span in el_spans if span.label_ == 'DIS'])

                doc.ents = tuple() # reset entities (setting new ones w.r.t. error type)
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

                if not self.nohtml:
                    self.generate_html(doc, f'./displacy/{article_id}_eval.html')

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(f'Precision = {precision}, Recall = {recall}')

    def generate_html(self, doc, filepath):
        """
        Generates an html file for visualizing entities in a proccessed document (.txt file).
        """

        if self.domain == 'diseases':
            ents = 'DIS'
        elif self.domain == 'food':
            ents = 'FOOD'

        html = displacy.render(doc, style='ent', page=True,
                             options={'colors': {'TP': '#00FF00', 'FN': '#FF0000', 'FP': '#FF00FF'},
                             'ents': [ents]})
        with open(filepath, 'w') as html_file:
            print(f'Saving a generated file to {filepath}')
            html_file.write(html)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', help='Specifies the task to perform (possible options: predict, evaluate).')
    parser.add_argument('domain', help='Specifies what to extract (possible options: diseases, food, relations).')
    parser.add_argument('datapath', help='Specifies a path to directory containing .txt files.')
    parser.add_argument('--nohtml', help='Use NOHTML=1 to disable generating html files with entities highlighted.',
                        default=0, type=int)
    parser.add_argument('--nolabeling', help='Use NOLABELING=1 to disable generating *_pred.txt files with labeled entities \
                        as a result of prediction', default=0, type=int)
    args = parser.parse_args()

    e = Extractor(args.task, args.domain, args.datapath, args.nohtml, args.nolabeling)
    e.run()
