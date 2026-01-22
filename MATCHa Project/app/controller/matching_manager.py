import numpy as np
from pprint import pprint

from paths import TESTDATA_DIR
from app.matching.matcher import get_matches
from app.core.embeddings.embedding import embed
from app.matching.similarity import st_sim, combine_two, combine_sims
from app.utils.input import read_headers, read_file
from app.core.string_based.sim_measures import lev_similarity, jaccard_sim
from app.core.metadata.metadata import find_equal_types, find_overlap

"""Currently mainly used for trialing and execution."""

file_name1 = TESTDATA_DIR / 'prelim_datasets' / 'RemoteWork_myConfigA1.csv'
file_name2 = TESTDATA_DIR / 'prelim_datasets' / 'RemoteWork_myConfigB1.csv'

df1_data = read_file(file_name1, ",")
df2_data = read_file(file_name2, ",")


# print(df1.dtypes)
# print("------------------------------------------------------")
# print(df2.dtypes)
# print("------------------------------------------------------")
# print(find_equal_types(df1, df2))

df1_header = read_headers(file_name1, ',')
df2_header = read_headers(file_name2, ',')

print(df1_header)
print(df2_header)

emb1 = embed(df1_header)
emb2 = embed(df2_header)

similarities_emb = st_sim(emb1, emb2)
print(similarities_emb)

similarities_lev = lev_similarity(df1_header, df2_header)
print(similarities_lev)

similarities_jac = jaccard_sim(df1_header, df2_header, 3)
print(similarities_jac)

similarities_num = find_overlap(df1_data, df2_data, 50)
print(similarities_num)

similarities_comp = combine_sims([similarities_emb, similarities_jac, similarities_lev], (0.65, 0.25, 0.1))
similarities_all = combine_sims([similarities_emb, similarities_jac, similarities_lev, similarities_num], (0.60, 0.20, 0.1, 0.1))
print("-------------------------------------------------")
print(similarities_all)

matches_comp = get_matches(df1_header, df2_header, similarities_comp, 0.6)
print(matches_comp)

# matches_emb = get_matches(df1_header, df2_header, similarities_emb, 0.6)
# print(matches_emb)

matches_all = get_matches(df1_header, df2_header, similarities_all, 0.6)
print(matches_all)

# matches_jac = get_matches(df1_header, df2_header, similarities_jac, 0.6)
# print(matches_jac)

