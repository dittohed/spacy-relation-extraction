import numpy as np
import argparse

import spacy
from spacy import displacy

from extractor import Extractor

class Pattern:
    """
    Represents a pattern (or cluster of patterns), where each pattern consists of:
    - left context embedding vector (average of word embeddings left to the left-most entity);
    - left-most entity label (FOOD / DIS);
    - middle context embedding vector (average of word embeddings between entities);
    - right-most entity label (FOOD / DIS);
    - right context embedding vector (average of word embeddings right to the right-most entity);
    - list of text representations of contributing patterns.
    """

    def __init__(self, left_vec, l_entity, middle_vec, r_entity, right_vec, text_rep):
        self.left_vec = left_vec
        self.l_entity = l_entity
        self.middle_vec = middle_vec
        self.r_entity = r_entity
        self.right_vec = right_vec
        self.contribs = text_rep

    def __str__(self):
        return ''.join([f'{contrib[0]} || {self.l_entity} || {contrib[1]} || {self.r_entity} ||  {contrib[2]}\n' \
                                                            for contrib in self.contribs])

    def is_entity_order_same(self, to_compare):
        """
        Returns True if both patterns have the same left-most entity label.
        """

        return self.l_entity == to_compare.l_entity

    def similarity(self, to_compare):
        if self.is_entity_order_same(to_compare):
            l_dot = np.dot(self.left_vec, to_compare.left_vec)
            m_dot = np.dot(self.middle_vec, to_compare.middle_vec)
            r_dot = np.dot(self.right_vec, to_compare.right_vec)

            similarity = l_dot + 2 * m_dot + r_dot

        else:
            similarity = 0

        print(f"Calculating similarity for: \n{self} \n and \n {to_compare}\n = {similarity}")
        return similarity

    def merge(self, to_merge):
        """
        Adds a new pattern to a cluster using cumulative moving average to
        limit the impact of the new pattern.
        """

        n = len(self.contribs) # no. of patterns in the cluster so far

        self.left_vec = (n*self.left_vec + to_merge.left_vec) / (n+1)
        self.middle_vec = (n*self.middle_vec + to_merge.middle_vec) / (n+1)
        self.right_vec = (n*self.right_vec + to_merge.right_vec) / (n+1)
        self.contribs += to_merge.contribs

class Snowball:
    def __init__(self, datapath, seed_tuples, n_iterations, w_size,
                 tau_cl, tau_supp, tau_sim, export_sents):
        """
        TODO
        """

        self.nlp = spacy.load("en_core_web_lg", disable=['ner'])
        # self.nlp.max_length = 2000000

        self.seed_tuples = seed_tuples # most common <DIS, FOOD> tuples extracted using relations_extractor
        self.tuples = seed_tuples # will contain all extracted tuples

        self.datapath = datapath

        self.patterns = [] # list of Pattern objects

        self.n_iterations = n_iterations
        self.w_size = w_size
        self.tau_cl = tau_cl
        self.tau_supp = tau_supp
        self.tau_sim = tau_sim

        self.export_sents = export_sents

        self.doc = None
        self.remove_irrel_sents()

    def remove_irrel_sents(self):
        """
        Sets a doc containing tagged sentences with both DIS and FOOD only.
        """

        # extracts labels for all articles
        e = Extractor('both', self.datapath, to_evaluate=0, nohtml=1, for_snowball=True)
        e.run()

        sents = ' '.join(e.sents) # new, long string with sentences containing both entitity types only

        self.doc = self.nlp(sents)
        e.extract_labels(self.doc) # extract DIS and FOOD labels once more (on account of spaCy's limitations)

        if self.export_sents:
            html = displacy.render(self.doc, style='ent', page=True,
                            options={'colors': {'DIS': '#909090', 'FOOD': '#19D9FF'}, 'ents': ['DIS', 'FOOD']})

            with open('./snowball_data/sents.html', 'w') as html_file:
                print('Saving extracted sents to snowball_data/sents.html')
                html_file.write(html)

    def run(self):
        for i in range(self.n_iterations):
            print(f'Iteration {i+1}/{self.n_iterations}')

            # find seed tuples occurences (order matters)
            for seed_tuple in self.tuples:
                for sent in self.doc.sents:
                    match_dis = None
                    match_food = None

                    for ent in sent.ents: # check for particular seed tuple occurence in the sentence
                        if seed_tuple[ent.label_] == ent.text:
                            if ent.label_ == 'DIS':
                                match_dis = ent
                            if ent.label_ == 'FOOD':
                                match_food = ent

                    if match_dis and match_food: # if sentence contains a seed tuple
                        print(f'Tuple: {seed_tuple.values} matched to sentence: {sent.text}')
                        if not seed_tuple['WAS_COUNTED']:
                            seed_tuple['N_OCCUR'] += 1

                        start = min(match_food.start, match_dis.start)
                        end = max(match_food.end, match_dis.end)

                        # compute contexts
                        left_len = min(self.w_size, start - sent.start)
                        right_len = min(self.w_size, sent.end - end)

                        ctx = self.doc[start - left_len : end + right_len]
                        left_ctx = ctx[0 : left_len]
                        mid_ctx = ctx[left_len : len(ctx) - right_len]
                        mid_ctx = self.remove_entities_from_middle_context(mid_ctx)
                        right_ctx = ctx[-right_len : ]

                        # calculate vector representations for contexts
                        left_vec = self.ctx2vec(left_ctx)
                        mid_vec = self.ctx2vec(mid_ctx)
                        right_vec = self.ctx2vec(right_ctx)

                        # get order of entities
                        entities_in_order = self.get_ents_order(match_dis, match_food)

                        pattern = Pattern(left_vec, entities_in_order[0].label_,
                                          mid_vec, entities_in_order[1].label_,
                                          right_vec, [(left_ctx.text, mid_ctx.text, right_ctx.text)])
                        print(f'Created a new pattern: \n{pattern}')
                        self.patterns.append(pattern)

            # sents = [
            #     ('consumption of', "doesn't cause", '.'),
            #     ('consuming', 'is the cause of', '.'),
            #     ('yellow bike', 'blue sea', 'crazy ramsay'),
            #     ('eating', 'causes', 'and other diseases'),
            # ]
            #
            # self.patterns = [
            #     Pattern(self.ctx2vec(self.nlp(sent[0])[:]), 'FOOD', self.ctx2vec(self.nlp(sent[1])[:]),
            #             'DIS', self.ctx2vec(self.nlp(sent[2])[:]), [sent]) \
            #                                 for sent in sents
            # ]
            #
            # self.patterns[3].l_entity = 'DIS'
            # self.patterns[3].r_entity = 'FOOD'

            # disable counting for current seed tuples
            for tuple in self.seed_tuples:
                tuple['WAS_COUNTED'] = True

            self.single_pass_clustering()
            self.drop_insufficient_clusters()

            for sent in self.doc.sents:
                match_dis = None
                match_food = None

                for ent in sent.ents:
                    if ent.label_ == 'DIS':
                        match_dis = ent
                    elif ent.label_ == 'FOOD':
                        match_food = ent

                if match_food is None or match_dis is None:
                    continue

                start = min(match_food.start, match_dis.start)
                end = max(match_food.end, match_dis.end)

                # compute contexts
                left_len = min(self.w_size, start - sent.start)
                right_len = min(self.w_size, sent.end - end)

                ctx = self.doc[start - left_len : end + right_len]
                left_ctx = ctx[0 : left_len]
                mid_ctx = ctx[left_len : len(ctx) - right_len]
                mid_ctx = self.remove_entities_from_middle_context(mid_ctx)
                right_ctx = ctx[-right_len : ]

                # calculate vector representations for contexts
                left_vec = self.ctx2vec(left_ctx)
                mid_vec = self.ctx2vec(mid_ctx)
                right_vec = self.ctx2vec(right_ctx)

                # get order of entities
                entities_in_order = self.get_ents_order(match_dis, match_food)

                pattern_candidate = Pattern(left_vec, entities_in_order[0].label_,
                                  mid_vec, entities_in_order[1].label_,
                                  right_vec, [(left_ctx.text, mid_ctx.text, right_ctx.text)])
                print(f'Created a new tuple candidate: {entities_in_order}')

                for i, pattern in enumerate(self.patterns):
                    print(f"Comparing \n{pattern} to current tuple.")

                    if pattern.similarity(pattern_candidate) > self.tau_sim:
                        tuple_candidate = {entities_in_order[0].label_: entities_in_order[0].text,
                                     entities_in_order[1].label_: entities_in_order[1].text,
                                     'N_OCCUR': 0, 'WAS_COUNTED': False}

                        if tuple_candidate not in self.tuples:
                            self.tuples.append(tuple_candidate)

                        break

    def ctx2vec(self, ctx):
        """
        Returns a vector representation for a given context (embeddings average).
        """

        if len(ctx) == 0:
            vec = np.zeros(300, dtype='float32')
        else:
            vec = ctx.vector / ctx.vector_norm

        return vec

    def single_pass_clustering(self):
        new_patterns = []

        for pattern in self.patterns:
            max_sim = 0
            max_sim_pattern = 0
            for i, new_pattern in enumerate(new_patterns):
                sim = pattern.similarity(new_pattern)

                if sim > max_sim:
                    max_sim = sim
                    max_sim_pattern = i

            if max_sim > self.tau_cl:
                new_patterns[max_sim_pattern].merge(pattern)
            else:
                new_patterns.append(pattern)

        self.patterns = new_patterns

    def remove_entities_from_middle_context(self, mid):
        extracted_mid = []
        for word in mid:
            if word.ent_type_ in ['DIS', 'FOOD']:
                pass
            else:
                extracted_mid.append(word.text)
        sub_doc = self.nlp(" ".join(extracted_mid))
        return sub_doc

    @staticmethod
    def get_ents_order(ent1, ent2):
        if ent1.start < ent2.start:
            return (ent1, ent2)
        else:
            return (ent2, ent1)

    def drop_insufficient_clusters(self):
        self.patterns = [pattern for pattern in self.patterns if len(pattern.contribs) >= self.tau_supp]

    def get_results(self):
        filepath = './snowball_data/results.txt'

        # sorting by occurences or contribs
        self.seed_tuples.sort(key=lambda d: d['N_OCCUR'], reverse=True)
        self.tuples.sort(key=lambda d: d['N_OCCUR'], reverse=True)
        self.patterns.sort(key=lambda p: p.contribs, reverse=True)

        with open(filepath, 'w') as file:
            print(f'Saving results to {filepath}')

            file.write('--- SEED TUPLES ---\n')
            [file.write(f'DIS: {tuple["DIS"]}, FOOD: {tuple["FOOD"]}, N_OCCUR: {tuple["N_OCCUR"]}\n') \
                                                                for tuple in self.seed_tuples]

            file.write('\n--- TUPLES ---\n')
            [file.write(f'DIS: {tuple["DIS"]}, FOOD: {tuple["FOOD"]}, N_OCCUR: {tuple["N_OCCUR"]}\n') \
                                                                for tuple in self.tuples]

            file.write('\n--- PATTERNS ---\n')
            [file.write(str(pattern) + '\n') for pattern in self.patterns]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath',
                        help='Specifies a path to directory containing .txt files.')
    parser.add_argument('--niterations',
                        help='Specifies number of iterations for snowball.',
                        default=3, type=int)
    parser.add_argument('--wsize',
                        help='Specifies a windows size for left and right contexts.',
                        default=3, type=int)
    parser.add_argument('--taucl',
                        help='Specifies a tau_cl parameter for thresholding patterns clustering (0 to 4).',
                        default=3, type=int)
    parser.add_argument('--tausupp',
                        help='Specifies a tau_supp parameter for minimal supporting tuples for a new pattern.',
                        default=3, type=int)
    parser.add_argument('--tau_sim',
                        help='Specifies a tau_sim parameter for minimal similarity with pattern to extract a new tuple (0 to 4).',
                        default=3, type=int)
    parser.add_argument('--exportsents',
                        help='Use --evaluate 1 to evaluate files with trailing _test.txt in name.',
                        default=0, type=int)

    args = parser.parse_args()

    snb = Snowball(
        args.datapath,
        [
        {'DIS': 'cancer', 'FOOD': 'diet', 'N_OCCUR': 0, 'WAS_COUNTED': False},
        {'DIS': 'cancer', 'FOOD': 'diet', 'N_OCCUR': 0, 'WAS_COUNTED': False},
        {'DIS': 'cancer', 'FOOD': 'diet', 'N_OCCUR': 0, 'WAS_COUNTED': False}],
        1, # args.niterations,
        args.wsize,
        args.taucl,
        args.tausupp,
        args.tausim,
        args.exportsents)

    snb.run()
    snb.get_results()
