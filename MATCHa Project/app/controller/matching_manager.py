import numpy as np
from pprint import pprint
from enum import Enum

from paths import TESTDATA_DIR
from app.matching.matcher import get_matches
from app.core.embeddings.embedding import embed, mean_decomp, pca_decomposition
from app.matching.similarity import st_sim, combine_two, combine_sims, cosine
from app.utils.input import read_headers, read_file
from app.core.string_based.sim_measures import lev_similarity, jaccard_sim
from app.core.metadata.metadata import find_equal_types, find_overlap

"""Currently mainly used for trialing and execution."""

class Matcher(Enum):
    JAC = 1
    LEV = 2
    EMB = 3
    EMB_MEAN = 4
    OVERLAP = 5
    DTYPES = 6

# TODO: add Overlap sim & Dtypes
def call_stuff(df1, df2, matchers, weights, threshold, prnt_sim = False, prnt_matches = False):
    sims = []
    emb1 = embed(df1)
    emb2 = embed(df2)
    for matcher in matchers:
        if matcher == Matcher.JAC:
            sims.append(jaccard_sim(df1, df2, 3))
        elif matcher == Matcher.LEV:
            sims.append(lev_similarity(df1, df2))
        elif matcher == Matcher.EMB:
            sims.append(cosine(emb1, emb2))
        elif matcher == Matcher.EMB_MEAN:
            e1, e2 = mean_decomp(emb1, emb2)
            sims.append(cosine(e1, e2))
    sim_matrix = combine_sims(sims, weights)
    if prnt_sim:
        print("Similarity matrix:")
        print(sim_matrix)
    matches = get_matches(df1, df2, sim_matrix, threshold)
    if prnt_matches:
        print("Matches:")
        print(matches)
    return sim_matrix, matches

file_name1 = TESTDATA_DIR / 'prelim_datasets' / 'steam_A1.csv'
file_name2 = TESTDATA_DIR / 'prelim_datasets' / 'steam_B1.csv'

df1_data = read_file(file_name1, ",")
df2_data = read_file(file_name2, ",")

print(find_overlap(df1_data, df2_data, 50))


# print(df1.dtypes)
# print("------------------------------------------------------")
# print(df2.dtypes)
# print("------------------------------------------------------")
# print(find_equal_types(df1, df2))

df1 = read_headers(file_name1, ',')
df2 = read_headers(file_name2, ',')

print(df1)
print(df2)

emb1 = embed(df1)
emb2 = embed(df2)

#call_stuff(df1, df2, [Matcher.JAC, Matcher.EMB], [0.4, 0.6], 0.6, False, True)
#call_stuff(df1, df2, [Matcher.JAC, Matcher.EMB_MEAN], [0.4, 0.6], 0.6, False, True)

# compare also with emb with numeric calcs:

sim_emb = cosine(emb1, emb2)
e1, e2 = mean_decomp(emb1, emb2)
sim_emb_mean = cosine(e1, e2)
sim_jac = jaccard_sim(df1, df2, 3)
sim_overlap = find_overlap(df1_data, df2_data, 0.5)

comb_ov_emb = combine_sims([sim_emb, sim_jac, sim_overlap], [0.6, 0.3, 0.1])
comb_ov_mean = combine_sims([sim_emb_mean, sim_jac, sim_overlap], [0.6, 0.3, 0.1])
comb_emb = combine_sims([sim_emb, sim_jac], [0.65, 0.35])
comb_mean = combine_sims([sim_emb_mean, sim_jac], [0.65, 0.35])

print("nn + nn:")
print(get_matches(df1, df2, comb_emb, 0.7))
print("nn + mn:")
print(get_matches(df1, df2, comb_mean, 0.7))
print("ol + nn:")
print(get_matches(df1, df2, comb_ov_emb, 0.7))
print("ol + mn:")
print(get_matches(df1, df2, comb_ov_mean, 0.7))
print("--------------------------------------------------")
print(get_matches(df1, df2, sim_emb, 0.7))
print(get_matches(df1, df2, sim_emb_mean, 0.7))

