# Snowball algorithm class

import numpy as np
import spacy

from extractor import Extractor

class Cluster:
    """
    Represents a pattern or cluster of patterns, where each pattern consists of:
    - left context embedding vector (average of word embeddings left to the left-most entity);
    - left-most entity label (FOOD / DIS);
    - middle context embedding vector (average of word embeddings between entities);
    - right-most entity label (FOOD / DIS);
    - right context embedding vector (average of word embeddings right to the right-most entity).
    """

    def __init__(self, left, entity1, middle, entity2, right, text_rep):
        self.left = left
        self.entity1 = entity1
        self.middle = middle
        self.entity2 = entity2
        self.right = right
        self.components = text_rep

    def __str__(self):
        return f"Whole context: ({self.left} || {self.entity1} || {self.middle} || {self.entity2} || {self.right})"

    def print_vectors(self):
        print("Left vector : ", self.left.vector)
        print("Middle vector : ", self.middle.vector)
        print("Right vector : ", self.right.vector)

    def is_entity_order_the_same(self, to_compare):
        if self.entity1 == to_compare.entity1 \
                and self.entity2 == to_compare.entity2:
            return True
        return False

    def compute_similarity(self, to_compare):
        if self.is_entity_order_the_same(to_compare):
            l_dot = np.dot(self.left, to_compare.left)

            m_dot = np.dot(self.middle, to_compare.middle)

            r_dot = np.dot(self.right, to_compare.right)

            summed_similarity = l_dot + 2 * m_dot + r_dot

            print("Summed up similarity: ", summed_similarity)

            return summed_similarity
        else:
            print("Summed up similarity: ", 0)
            return 0

    def merge(self, to_merge):
        self.left = self.left + to_merge.left / 2
        self.middle = self.middle + to_merge.middle / 2
        self.right = self.right + to_merge.right / 2
        self.components.append(to_merge.components)


class Snowball:
    def __init__(self, tuples, datapath, number_of_iterations=1, tau_sim=3.5):
        self.large_nlp = spacy.load("en_core_web_lg")
        self.seed_tuples = tuples  # manually created seed tuples (DIS, FOOD)
        self.tuples = tuples
        self.rules = []  # list of contexts with rules and entities
        self.datapath = datapath
        self.matches = []
        self.tagged_doc = self.tag_entities()
        self.number_of_iterations = number_of_iterations
        self.tau_sim = tau_sim

    def tag_entities(self):
        # method used to tag important entities
        e = Extractor('both', self.datapath, 0, 1)
        e.run()
        return e.doc

    def run(self):
        # actual snowball algorithm
        # First analyze all sentences, find all containing both DIS(disease) and FOOD
        # Check if entities in sentence match any tuple, if yes extract context and the rule
        # If not check if context around matches any existing rule, if yes, extract new seed tuple
        while self.number_of_iterations > 0:
            self.number_of_iterations -= 1

            for sent in self.tagged_doc.sents:
                # sprawdzenie czy zdanie ma oba entity
                has_food = False
                has_disease = False
                for entity in sent.ents:
                    if entity.label_ == 'FOOD':
                        has_food = True
                    if entity.label_ == 'DIS':
                        has_disease = True

                # jeżęli ma oba entity
                if has_food and has_disease:
                    # Sentence is subject to analysis for a rule or tuple
                    is_match = False
                    for seed_tuple in self.tuples: # dla każdej seed-krotki
                        # Check if seed tuple is in the sentence AND is in the correct order
                        match_dis = False
                        match_food = False
                        ordered_keys = list(seed_tuple.keys())  # Entites from seed tuple in right order
                        for ent in sent.ents: # sprawdzenie, czy w zdaniu występuje seed krotka
                            if seed_tuple[ent.label_] == ent.text:
                                if ent.label_ == 'DIS':
                                    match_dis = ent
                                if ent.label_ == 'FOOD':
                                    match_food = ent
                        if match_food and match_dis: # jeżeli w zdaniu występuje seed krotka
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
        new_rules = []
        for cluster in self.rules:
            if len(new_rules) == 0:
                new_rules.append(cluster)
                continue
            for i, new_cluster in enumerate(new_rules):
                if cluster.compute_similarity(new_cluster) < TAU_CL:
                    new_rules.append(cluster)
                else:
                    new_rules[i].merge(cluster)
        self.rules = new_rules

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
        self.rules = [cluster for cluster in self.rules if len(rule.components) < TAU_SUPP]


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
