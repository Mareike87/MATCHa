import numpy as np

from paths import TESTDATA_DIR
from app.matching.matcher import get_matches
from app.core.embeddings.embedding import embed
from app.matching.similarity import st_sim, combine_two, combine_sims
from app.utils.input import read_headers, read_file
from app.core.string_based.sim_measures import lev_similarity, jaccard_sim
from app.core.metadata.metadata import find_equal_types

"""Currently mainly used for trialing and execution."""


file_name1 = TESTDATA_DIR / 'prelim_datasets' / 'diabetes_A1.csv'
file_name2 = TESTDATA_DIR / 'prelim_datasets' / 'diabetes_B1.csv'

# df1 = read_file(file_name1, ",")
# df2 = read_file(file_name2, ",")
#
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

similarities1 = st_sim(emb1, emb2)
print(similarities1)

# similarities2 = lev_similarity(df1, df2)
# print(similarities2)

similarities2 = jaccard_sim(df1, df2, 3)
print(similarities2)

similarities = combine_two(similarities1, similarities2, (0.7, 0.3))
similarities_new = combine_sims([similarities1, similarities2], (0.7, 0.3))
print(similarities)
print("-------------------------------------------------")
print(similarities_new)

matches = get_matches(df1, df2, similarities, 0.6)
print(matches)

matches_emb = get_matches(df1, df2, similarities1, 0.6)
print(matches_emb)
matches_jac = get_matches(df1, df2, similarities2, 0.6)
print(matches_jac)


# Takes two files, reads their headers and will calculate matches based on
# Jaccard and embeddings.
# Only for ease of trials, not to be used later (weights & thresholds need to
# be set manually)
def file_to_match_headers(file_name1, file_name2):
    df1 = read_headers(file_name1, ',')
    df2 = read_headers(file_name2, ',')
    print(df1)
    print(df2)
    emb1 = embed(df1)
    emb2 = embed(df2)
    similarities1 = st_sim(emb1, emb2)
    similarities2 = jaccard_sim(df1, df2, 3)
    similarities = combine_two(similarities1, similarities2, (0.7, 0.3))
    matches = get_matches(df1, df2, similarities, 0.6)
    return matches