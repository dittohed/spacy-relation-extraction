# Snowball algorithm class

import numpy as np
import spacy

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

    def __init__(self, left, l_entity, middle, r_entity, right, text_rep):
        self.left = left
        self.l_entity = l_entity
        self.middle = middle
        self.r_entity = r_entity
        self.right = right
        self.contribs = text_rep

    # def __str__(self):
    #     return f"Whole context: ({self.left} || {self.l_entity} || {self.middle} || {self.r_entity} || {self.right})"

    def is_entity_order_same(self, to_compare):
        """
        Returns True if both patterns have the same left-most entity label.
        """

        return self.l_entity == to_compare.l_entity

    def similarity(self, to_compare):
        if self.is_entity_order_same(to_compare):
            l_dot = np.dot(self.left, to_compare.left)
            m_dot = np.dot(self.middle, to_compare.middle)
            r_dot = np.dot(self.right, to_compare.right)

            similarity = l_dot + 2 * m_dot + r_dot

        else:
            similarity = 0

        # print("Weighted similarity: ", similarity)
        return similarity

    def merge(self, to_merge):
        """
        Adds a new pattern to a cluster using cumulative moving average to
        limit the impact of the new pattern.
        """

        n = len(self.contribs) # no. of patterns in the cluster so far

        self.left = (n*self.left + to_merge.left) / (n+1)
        self.middle = (n*self.middle + to_merge.middle) / (n+1)
        self.right = (n*self.right + to_merge.right) / (n+1)
        self.contribs.append(to_merge.contribs)

class Snowball:
    def __init__(self, tuples, datapath, n_iterations=1, w_size=3,
                 tau_cl=2, tau_supp=1, tau_sim=3.5, tau_t=0.5, alpha=0.5,
                 export_sents=False):
        """
        TODO
        """

        self.nlp = spacy.load("en_core_web_lg")

        self.seed_tuples = tuples # most common <DIS, FOOD> tuples extracted using relations_extractor
        self.tuples = tuples # ???

        self.datapath = datapath

        self.patterns = [] # list of Pattern objects
        self.matches = [] # ???

        self.number_of_iterations = number_of_iterations
        self.w_size = w_size
        self.tau_cl = tau_cl
        self.tau_supp = tau_supp
        self.tau_sim = tau_sim
        self.tau_t = tau_t
        self.alpha = alpha

        self.sents = self.get_sents_crops()

    def get_sents_crops(self):
        """
        Returns a list of crops with corresponding windows size (w_size)
        of sentences only for sentences with both DIS and FOOD.
        """

        e = Extractor('both', self.datapath, 0, 1)
        e.run()

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
                    sents.append(sent)
                    continue

        if self.export_sents:
            html = displacy.render(doc, style='ent', page=True,
                            options={'colors': {'DIS': '#909090', 'FOOD': '#19D9FF'}, 'ents': ['DIS', 'FOOD']})

            with open('./snowball_data/sents.html', 'w') as html_file:
                print('Saving extracted sents to snowball_data/sents.html')
                html_file.write(html)

    def run(self):
        # actual snowball algorithm
        # First analyze all sentences, find all containing both DIS(disease) and FOOD
        # Check if entities in sentence match any tuple, if yes extract context and the rule
        # If not check if context around matches any existing rule, if yes, extract new seed tuple

        while self.number_of_iterations > 0:
            self.number_of_iterations -= 1
            for sent in self.tagged_doc.sents:
                has_food = False
                has_disease = False
                for entity in sent.ents:
                    if entity.label_ == 'FOOD':
                        has_food = True
                    if entity.label_ == 'DIS':
                        has_disease = True
                if has_food and has_disease:
                    # Sentence is subject to analysis for a rule or tuple
                    is_match = False
                    for seed_tuple in self.tuples:
                        # Check if seed tuple is in the sentence AND is in the correct order
                        match_dis = False
                        match_food = False
                        ordered_keys = list(seed_tuple.keys())  # Entites from seed tuple in right order
                        for ent in sent.ents:
                            if seed_tuple[ent.label_] == ent.text:
                                if ent.label_ == 'DIS':
                                    match_dis = ent
                                if ent.label_ == 'FOOD':
                                    match_food = ent
                        if match_food and match_dis:
                            entities_in_order = self.select_order(match_dis, match_food)
                            is_match = self.are_in_order(ordered_keys,
                                                         [entities_in_order[0].label_, entities_in_order[1].label_])
                            if is_match:
                                print(f"Tuple {seed_tuple} Matched to sentence: {sent}")
                                start = min(match_food.start, match_dis.start)
                                end = max(match_food.end, match_dis.end)

                                context = self.tagged_doc[
                                          start - min(3, start - sent.start):end + min(3, sent.end - end)]

                                left_len = min(3, start - sent.start)
                                right_len = min(3, sent.end - end)

                                mid_ctx = context[left_len:len(context) - right_len]
                                mid_ctx = self.remove_entities_from_middle_context(mid_ctx)

                                left_ctx = context[0:left_len]
                                right_ctx = context[-right_len:]
                                # Check if rigth left and middle contexts exist
                                if len(left_ctx) == 0:
                                    left = 0
                                else:
                                    left = left_ctx.vector / left_ctx.vector_norm
                                if len(mid_ctx) == 0:
                                    mid = 0
                                else:
                                    mid = mid_ctx.vector / mid_ctx.vector_norm
                                if len(right_ctx) == 0:
                                    right = 0
                                else:
                                    right = right_ctx.vector / right_ctx.vector_norm

                                ctx = Cluster(
                                    left,
                                    entities_in_order[0].label_,
                                    mid,
                                    entities_in_order[1].label_,
                                    right,
                                    [(left_ctx.text, mid_ctx.text, right_ctx.text)])
                                print(f'Appending context: {ctx}')
                                self.rules.append(ctx)
                                # break

            self.single_pass_clustering()
            self.drop_insufficient_clusters()

            for sent in self.tagged_doc:
                disease = None
                food = None

                for ent in sent.ents:
                    if ent.label_ == 'DIS':
                        disease = ent
                    elif ent.label_ == 'FOOD':
                        food = ent

                if food and disease:
                    start = min(food.start, disease.start)
                    end = max(food.end, disease.end)

                    entities_in_order = self.select_order(disease, food)

                    context = self.tagged_doc[start - min(3, start - sent.start):end + min(3, sent.end - end)]

                    left_len = min(3, start - sent.start)
                    right_len = min(3, sent.end - end)

                    mid = context[left_len:len(context) - right_len]
                    mid = self.remove_entities_from_middle_context(mid)

                    left = context[0:left_len]
                    right = context[-right_len:]

                    if len(left_ctx) == 0:
                        left = 0
                    else:
                        left = left_ctx.vector / left_ctx.vector_norm
                    if len(mid_ctx) == 0:
                        mid = 0
                    else:
                        mid = mid_ctx.vector / mid_ctx.vector_norm
                    if len(right_ctx) == 0:
                        right = 0
                    else:
                        right = right_ctx.vector / right_ctx.vector_norm

                    ctx = Cluster(
                        left,
                        entities_in_order[0].label_,
                        mid,
                        entities_in_order[1].label_,
                        right
                        [left_ctx.text, mid_ctx.text, right_ctx.text])

                    for cluster in self.rules:
                        print(f"Comparing \n{ctx} to: ")
                        print(rule, '\n')
                        if ctx.compute_similarity(rule) > TAU_SIM:
                            new_tuple = {entities_in_order[0].label_: entities_in_order[0].text,
                                         entities_in_order[1].label_: entities_in_order[1].text}
                            if new_tuple not in self.tuples:
                                self.tuples.append(new_tuple)
                            break

    def single_pass_clustering(self):
        new_patterns = []

        for pattern in self.patterns:
            if len(new_patterns) == 0:
                new_patterns.append(pattern)
                continue

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

        self.pattern = new_patterns

    def remove_entities_from_middle_context(self, mid):
        extracted_mid = []
        for word in mid:
            if word.ent_type_ in ['DIS', 'FOOD']:
                pass
            else:
                extracted_mid.append(word.text)
        sub_doc = self.large_nlp(" ".join(extracted_mid))
        return sub_doc

    @staticmethod
    def select_order(ent1, ent2):
        if ent1.start < ent2.start:
            return [ent1, ent2]
        else:
            return [ent2, ent1]

    @staticmethod
    def are_in_order(l1, l2):
        for i in range(2):
            if l1[i] != l2[i]:
                return False
        return True

    def drop_insufficient_clusters(self):
        self.patterns = [pattern for pattern in self.patterns if len(rule.contribs) < self.tau_supp]


if __name__ == "__main__":
    snb = Snowball([
        {'DIS': 'DASH', 'FOOD': 'accordant diet'},
        {'DIS': 'Liver disease', 'FOOD': 'fat diary products'},
        {'DIS': 'lung cancer', 'FOOD': 'sausage'},
        {'FOOD': 'steak', 'DIS': 'flu'},
        {'FOOD': 'carrot', 'DIS': 'blindness'}
    ], './articles/snowball', 3, 3)
    snb.run()
    for rule in snb.rules:
        print(rule)
    for tuple in snb.tuples:
        print(tuple)
