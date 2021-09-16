# Snowball algorithm class
# NIE ROZRÓŻNIAMY NEGACJI

import numpy as np

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
    def __init__(self, tuples, datapath, n_iterations=5, w_size=3,
                 tau_cl=2, tau_supp=2, tau_sim=2.5, export_sents=False):
        """
        TODO
        """

        self.nlp = spacy.load("en_core_web_lg")

        self.seed_tuples = tuples # most common <DIS, FOOD> tuples extracted using relations_extractor
        self.tuples = tuples # will contain all extracted tuples

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
        e = Extractor('both', self.datapath, 0, 1)
        e.run()

        # finds sentences with both DIS and FOOD
        sents = []
        for doc in e.docs:
            for sent in doc.sents:
                has_food = False
                has_disease = False

                for entity in sent.ents:
                    if entity.label_ == 'FOOD':
                        has_food = True
                    if entity.label_ == 'DIS':
                        has_disease = True

                if has_food and has_disease:
                    sents.append(sent.text)
                    continue

        sents = ' '.join(sents) # new, long string with sentences containing both entitity types only
        self.doc = self.nlp(sents)
        e.extract_labels(self.doc) # extract DIS and FOOD labels once more (on account of spaCy's limitations)

        if self.export_sents:
            html = displacy.render(self.doc, style='ent', page=True,
                            options={'colors': {'DIS': '#909090', 'FOOD': '#19D9FF'}, 'ents': ['DIS', 'FOOD']})

            with open('./snowball_data/sents.html', 'w') as html_file:
                print('Saving extracted sents to snowball_data/sents.html')
                html_file.write(html)

    def run(self):
        # actual snowball algorithm
        # First analyze all sentences, find all containing both DIS(disease) and FOOD
        # Check if entities in sentence match any tuple, if yes extract context and the rule
        # If not check if context around matches any existing rule, if yes, extract new seed tuple

        for i in range(self.n_iterations):

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
                                     entities_in_order[1].label_: entities_in_order[1].text}

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

    # @staticmethod
    # def are_in_order(l1, l2):
    #     for i in range(2):
    #         if l1[i] != l2[i]:
    #             return False
    #     return True

    def drop_insufficient_clusters(self):
        self.patterns = [pattern for pattern in self.patterns if len(pattern.contribs) >= self.tau_supp]

    def get_results(self):
        # ostatecznie pozliczać i posortować
        filepath = './snowball_data/results.txt'

        with open(filepath, 'w') as file:
            print(f'Saving results to {filepath}')

            file.write('--- SEED TUPLES ---\n')
            [file.write(f'DIS: {tuple["DIS"]}, FOOD: {tuple["FOOD"]}\n') for tuple in self.seed_tuples]

            file.write('\n--- TUPLES ---\n')
            [file.write(f'DIS: {tuple["DIS"]}, FOOD: {tuple["FOOD"]}\n') for tuple in self.tuples]

            file.write('\n--- PATTERNS ---\n')
            [file.write(str(pattern) + '\n') for pattern in self.patterns]

if __name__ == "__main__":
    snb = Snowball([
        {'DIS': 'DASH', 'FOOD': 'accordant diet'},
        {'DIS': 'Liver disease', 'FOOD': 'fat diary products'},
        {'DIS': 'lung cancer', 'FOOD': 'sausage'},
        {'FOOD': 'steak', 'DIS': 'flu'},
        {'FOOD': 'carrot', 'DIS': 'blindness'}], './articles/relations', export_sents=True)

    snb.run()
    snb.get_results()

    # for pattern in snb.patterns:
    #     print(pattern)
    # for tuple in snb.tuples:
    #     print(tuple)
