import numpy as np

from paths import TESTDATA_DIR
from app.matching.matcher import get_matches
from app.core.embeddings.embedding import embed
from app.matching.similarity import st_sim
from app.utils.input import read_headers, read_file


file_name1 = TESTDATA_DIR / 'prelim_datasets' / 'diabetes_A1.csv'
file_name2 = TESTDATA_DIR / 'prelim_datasets' / 'diabetes_B1.csv'

df1 = read_headers(file_name1, ',')
df2 = read_headers(file_name2, ',')

print(df1)
print(df2)

emb1 = embed(df1)
emb2 = embed(df2)

similarities = st_sim(emb1, emb2)
print(similarities)

matches = get_matches(df1, df2, similarities, 0.7)
print(matches)