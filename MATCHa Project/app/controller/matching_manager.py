import numpy as np
from pprint import pprint
from enum import Enum

from paths import TESTDATA_DIR
from app.matching.matcher import get_matches
from app.core.embeddings.embedding import embed, mean_decomp
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

file_name1 = TESTDATA_DIR / 'prelim_datasets' / 'diabetes_A1.csv'
file_name2 = TESTDATA_DIR / 'prelim_datasets' / 'diabetes_B1.csv'

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

print(type(emb1))

similarities_emb = cosine(emb1, emb2)
#print(similarities_emb)

similarities_mean = cosine(mean_decomp(emb1,emb2)[0], mean_decomp(emb1,emb2)[1])
#print(similarities_mean)

matches_emb = get_matches(df1, df2, similarities_emb, 0.6)
print(matches_emb)

matches_mean = get_matches(df1, df2, similarities_mean, 0.6)
print(matches_mean)

similarities_lev = lev_similarity(df1, df2)
#print(similarities_lev)

similarities_jac = jaccard_sim(df1, df2, 3)
#print(similarities_jac)

similarities_all = combine_sims([similarities_emb, similarities_jac, similarities_lev], (0.65, 0.25, 0.1))
print("-------------------------------------------------")
#print(similarities_all)

matches = get_matches(df1, df2, similarities_all, 0.6)
print(matches)

similarities_all = combine_sims([similarities_mean, similarities_jac, similarities_lev], (0.65, 0.25, 0.1))
matches = get_matches(df1, df2, similarities_all, 0.6)
print(matches)

# TODO: add Overlap sim & Dtypes
def call_stuff(df1, df2, matchers, weights, threshold, prnt_sim = False, prnt_matches = False):
    sims = []
    for matcher in matchers:
        if matcher == Matcher.JAC:
            sims.append(jaccard_sim(df1, df2))
        elif matcher == Matcher.LEV:
            sims.append(lev_similarity(df1, df2))
        elif matcher == Matcher.EMB:
            sims.append(cosine(df1, df2))
        elif matcher == Matcher.EMB_MEAN:
            sims.append(mean_decomp(df1, df2))
    sim_matrix = combine_sims(sims, weights)
    if prnt_sim:
        print("Similarity matrix:")
        print(sim_matrix)
    matches = get_matches(df1, df2, sim_matrix, threshold)
    if prnt_matches:
        print("Matches:")
        print(matches)
    return sim_matrix, matches