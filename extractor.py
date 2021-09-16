# evaluates or predicts using specified extractor
# example of usage: python extractor.py diseases ./articles/diseases --evaluate 1

import os

import argparse
import glob
import re
import copy

import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher, DependencyMatcher
from spacy import displacy
from spacy.util import filter_spans

import diseases_extractor as dis
import food_extractor as food
import relations_extractor as rel

class Extractor:
    """
    True labels should be denoted with enclosing semi-colons exactly this way:
    ;;<token>;;
    or (in case of multiple-token entity):
    ;;<token1> ... <tokenN>;;
    """

    TL_EXPRESSION = ';;.*?;;' # non-greedy regexp for identifying true labels

    def __init__(self, domain, datapath, to_evaluate, nohtml, for_snowball=False):
        self.domain = domain
        self.datapath = datapath
        self.to_evaluate = to_evaluate
        self.nohtml = nohtml

        self.nlp = spacy.load('en_core_web_lg', disable=['ner'])
        # self.nlp.max_length = 2000000

        # snowball
        self.for_snowball = for_snowball
        self.sents = [] # list of sentences with both DIS and FOOD from all articles

    def run(self):
        if self.to_evaluate:
            self.evaluate()
        else:
            self.predict()

    def extract_labels(self, doc):
        """
        Extract labels using a proper extractor.
        """

        matcher = Matcher(self.nlp.vocab)
        matcher_dep = DependencyMatcher(self.nlp.vocab)

        doc.ents = tuple()

        if self.domain == 'diseases':
            matcher_dep.add('dependencies', dis.dependencies_patterns,
                        on_match=dis.add_disease_ent_dep)
            matcher_dep(doc)

        elif self.domain == 'food':
            # doc.ents = tuple([ent for ent in doc.ents if ent.label_ in ('DIS')])

            matcher_dep.add('dependencies', food.dependencies_patterns,
                        on_match=food.add_food_dep)
            matcher_dep(doc)

            food.merge_entities(doc)

        elif self.domain == 'both':
            matcher_dep.add('dependencies_dis', dis.dependencies_patterns,
                        on_match=dis.add_disease_ent_dep)
            matcher_dep.add('dependencies_food', food.dependencies_patterns,
                        on_match=food.add_food_dep)
            matcher_dep(doc)

            food.merge_entities(doc)

        else:
            matcher_dep.add('dependencies_dis', dis.dependencies_patterns,
                        on_match=dis.add_disease_ent_dep)
            matcher_dep.add('dependencies_food', food.dependencies_patterns,
                        on_match=food.add_food_dep)

            matcher_dep(doc)

            matcher.add('associations', rel.association_patterns,
                        on_match=rel.add_associations_ent)

            matcher(doc)

            temp_doc = copy.deepcopy(doc)

            matcher_dep.remove('dependencies_dis')
            matcher_dep.remove('dependencies_food')

            matcher_dep.add('dependencies_rel', rel.relations_patterns,
                        on_match=rel.add_relations_ent_dep)
            matcher_dep(doc)

            food.merge_entities(doc)

            self.relations_data = rel.extract_relations_data(temp_doc, doc)

    def predict(self):
        """
        Predicts entities.
        """

        articles = glob.glob(f'{self.datapath}/*.txt')

        for article in articles:
            if 'test.txt' in article:
                continue

            with open(article, 'r') as file:
                article_id = article.split('/')[-1].split('.')[0]
                print(f'Handling article {article_id}...')

                # if os.path.exists(f'./displacy/{self.domain}/{article_id}_pred.html'):
                #     print(f'{article_id} already predicted...')
                #     continue

                lines = file.readlines()
                text = ' '.join(lines)

                # checking for memory requirements
                if len(text) > 1000000:
                    print(f'Article {article_id} is too long, skipping to the next one...')
                    continue

                doc = self.nlp(text)
                self.extract_labels(doc)

                # snowball
                if self.for_snowball:
                    for sent in doc.sents:
                        has_food = False
                        has_disease = False

                        for entity in sent.ents:
                            if entity.label_ == 'FOOD':
                                has_food = True
                            if entity.label_ == 'DIS':
                                has_disease = True

                        if has_food and has_disease:
                            self.sents.append(sent.text)

                if not self.nohtml:
                    self.generate_html(doc, f'./displacy/{self.domain}/{article_id}_pred.html')

                if self.domain == 'relations':
                    print('--- EXTRACTED RELATIONS: ---')
                    print(self.relations_data)

                    self.save_relations_data(f'./relations_data/{article_id}.txt')

    def evaluate(self):
        """
        Evaluates extraction method by comparing labeled (TL_EXPRESSION) .txt files
        and extracted labels.
        Prints out precision and recall.
        WARNING: takes files ending with _test in account only!
        """

        tp = 0 # true positive
        fp = 0 # false positive
        fn = 0 # false negative

        articles = glob.glob(f'{self.datapath}/*_test.txt')

        if self.domain == 'diseases':
            label = 'DIS'
        elif self.domain == 'food':
            label = 'FOOD'
        else:
            pass

        for article in articles:
            with open(article, 'r') as file:
                article_id = article.split('/')[-1].split('.')[0]
                print(f'Handling article {article_id}...')

                if os.path.exists(f'./displacy/{self.domain}/{article_id}_eval.html'):
                    print(f'{article_id} already evaluated...')
                    continue

                lines = file.readlines()
                text = ' '.join(lines)

                # checking for memory requirements
                if len(text) > 1000000:
                    print(f'Article {article_id} is too long, skipping to the next one...')
                    continue

                # reading true labels and creating Span objects
                tl_found = 0
                tl_positions = []
                for match in re.finditer(self.TL_EXPRESSION, text): # matches are returned in left-to-right order
                    start, end = match.span()

                    tl_positions.append((start-(tl_found)*4, end-(tl_found+1)*4))
                    tl_found += 1

                doc = self.nlp(text.replace(';;', '')) # labels not necessary anymore
                tl_spans = tuple([doc.char_span(start, end, label=label, alignment_mode='expand') \
                                for (start, end) in tl_positions])

                self.extract_labels(doc)

                el_spans = doc.ents # extracted labels (Span objects)

                # comparing true labels with extracted labels
                tl_spans = set([(span.start, span.end) for span in tl_spans])
                el_spans = set([(span.start, span.end) for span in el_spans if span.label_ == label])

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
                    self.generate_html(doc, f'./displacy/{self.domain}/{article_id}_eval.html')

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        print(f'Precision = {precision}, Recall = {recall}')

    def generate_html(self, doc, filepath):
        """
        Generates an html file for visualizing entities in a proccessed document (.txt file).
        """

        if self.to_evaluate:
            ents = ['TP', 'FN', 'FP']
        elif self.domain == 'diseases':
            ents = ['DIS']
        elif self.domain == 'food':
            ents = ['FOOD']
        elif self.domain == 'both':
            ents = ['DIS', 'FOOD']
        elif self.domain == 'relations':
            ents = ['REL']

        html = displacy.render(doc, style='ent', page=True,
                             options={'colors': {'TP': '#00FF00', 'FN': '#FF0000', 'FP': '#FF00FF',
                                                 'DIS': '#909090', 'FOOD': '#19D9FF', 'REL': '#0064FF'}, 'ents': ents})

        with open(filepath, 'w') as html_file:
            print(f'Saving a generated file to {filepath}')
            html_file.write(html)

    def save_relations_data(self, filepath):
        with open(filepath, 'w') as relations_file:
            print(f'Saving relations data to {filepath}')
            relations_file.write(self.relations_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('domain', help='Specifies what to extract (possible options: diseases, food, both, relations).',
                        choices=['diseases', 'food', 'both', 'relations'])
    parser.add_argument('datapath', help='Specifies a path to directory containing .txt files.')
    parser.add_argument('--evaluate', help='Use --evaluate 1 to evaluate files with trailing _test.txt in name.',
                        default=0, type=int)
    parser.add_argument('--nohtml', help='Use --nohtml 1 to disable generating html files with entities highlighted.',
                        default=0, type=int)

    args = parser.parse_args()

    e = Extractor(args.domain, args.datapath, args.evaluate, args.nohtml)
    e.run()
    # you can use e.doc after calling e.run()
