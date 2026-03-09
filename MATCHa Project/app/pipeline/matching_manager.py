from enum import Enum

import pandas as pd

from app.similarity.instance.top_k import top_k_sim
from app.similarity.schema.type import find_type_similarity
from paths import TESTDATA_DIR
from app.matching.matcher import get_matches
from app.similarity.schema.embedding import embed, mean_decomp
from app.similarity.aggregation import combine_sims_weighted, combine_sims_var
from app.similarity.schema.embedding import cosine
from app.utils.input import read_headers, read_file
from app.similarity.schema.string import lev_similarity, jaccard_sim
#from app.similarity.schema.type import find_equal_types
from app.similarity.instance.numerical import find_overlap

"""Currently mainly used for trialing and execution."""

# diabetesA = TESTDATA_DIR / 'final_datasets' / 'diabetes' / 'diabetes_lv3_A.csv'
# diabetesB = TESTDATA_DIR / 'final_datasets' / 'diabetes' / 'diabetes_lv3_B.csv'
# gt_diabetes = TESTDATA_DIR / 'ground_truth' / 'diabetes_lv3_map.csv'
#
# gymA = TESTDATA_DIR / 'final_datasets' / 'gym_members' / 'gym_lv3_A.csv'
# gymB = TESTDATA_DIR / 'final_datasets' / 'gym_members' / 'gym_lv3_B.csv'
# gt_gym = TESTDATA_DIR / 'ground_truth' / 'gym_lv3_map.csv'
#
# steamA = TESTDATA_DIR / 'final_datasets' / 'steam' / 'steam_lv1_A.csv'
# steamB = TESTDATA_DIR / 'final_datasets' / 'steam' / 'steam_lv1_B.csv'
# gt_steam = TESTDATA_DIR / 'ground_truth' / 'steam_lv1_map.csv'

def run_matching(datapath1, datapath2, delimiter, threshold,schema=True, instance=True):
    similarities = []
    masks = []
    headers1 = read_headers(datapath1, delimiter)
    headers2 = read_headers(datapath2, delimiter)
    if schema:  # caution: sollte hier nur embeddings || mean embed?
        # embeddings:
        emb1 = embed(headers1)
        emb2 = embed(headers2)
        sim, mask = cosine(emb1, emb2)
        similarities.append(sim)
        masks.append(mask)
        # embeddings + mean
        emb1, emb2 = mean_decomp(emb1, emb2)
        sim, mask = cosine(emb1, emb2)
        similarities.append(sim)
        masks.append(mask)
        # jaccard
        sim, mask = jaccard_sim(headers1, headers2, 3)  # 3 or differen? adjustment opportunity
        similarities.append(sim)
        masks.append(mask)
        # levenshtein
        sim, mask = lev_similarity(headers1, headers2)
        similarities.append(sim)
        masks.append(mask)
    if instance:
        df1 = read_file(datapath1, delimiter)
        df2 = read_file(datapath2, delimiter)
        # type check
        sim, mask = find_type_similarity(df1, df2)
        similarities.append(sim)
        masks.append(mask)
        # overlap percentile
        sim, mask = find_overlap(df1, df2, 50)
        similarities.append(sim)
        masks.append(mask)
        # top k
        sim, mask = top_k_sim(df1, df2, 50)
        similarities.append(sim)
        masks.append(mask)
    sim_final = combine_sims_var(similarities, masks)
    matches = get_matches(headers1, headers2, sim_final, threshold)
    return matches




# CURRENT PROBLEM: Type checks sind nicht nuanciert genug -> führen zu hoher Varianz -> werden zu hoch gewichtet. Das ist schlecht.