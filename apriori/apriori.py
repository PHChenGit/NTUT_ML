import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from typing import Optional, List

df = pd.read_csv("./dataset/apriori_training.csv")
categories = list(df.columns)
total_trans = int(df.shape[0])


def preprocess_data() -> List[set]:
    _, training_data, verify_data = np.split(df, [0, 800])

    selected_items_list = []
    for idx, row in training_data.iterrows():
        selected_items = []
        selected_item_set = set()

        for item in categories:
            if row[item]:
                selected_items.append(item)
                selected_item_set.add(item)
        selected_items_list.append(selected_item_set)

    return selected_items_list


def apriori(min_support, min_confidence):
    """

    :param min_support:
    :param min_confidence:
    :return:
    """

    def generate_candidates(data: set, size: int) -> List[set]:
        res = []

        data = sorted(data)
        item_set_list = [item for item in data]
        # for item in data:
        #     item_set = set()
        #     item_set.add(item)
        #     item_set_list.append(item_set)

        for i in range(len(item_set_list)):
            current_set = item_set_list[i]
            for j in range(i + 1, len(data)):
                current_set = current_set | item_set_list[j]
                if len(current_set) == size:
                    res.append(current_set)
                    current_set = item_set_list[i]

        return res

    def get_L_k_list(data: List[set], source_data: List[set]) -> set:
        item_counter = dict()

        for item_set in data:
            cnt = 0
            for trans in source_data:
                if item_set & trans:
                    cnt += 1

            item_counter.update({item_set: cnt})

        L_k = set()

        for item, sup in item_counter.items():
            support = sup / total_trans
            if support > min_support:
                L_k.add(item)

        return L_k

    """
    Step 2-1: Generate C_k and L_k
    """
    confirm_transactions = preprocess_data()
    C_1 = [frozenset([category]) for category in categories]
    L_1 = get_L_k_list(C_1, confirm_transactions)
    L_list = [L_1]
    k = 2

    while L_list[k - 2]:
        new_item_set = generate_candidates(L_list[k - 2], k)
        L_k = get_L_k_list(new_item_set, confirm_transactions)
        L_list.append(L_k)
        k += 1

    """
    Step 3: Generate association rules
    """



if __name__ == "__main__":
    min_support = 0.28
    min_confidence = 0.4

    rules = apriori(min_support, min_confidence)