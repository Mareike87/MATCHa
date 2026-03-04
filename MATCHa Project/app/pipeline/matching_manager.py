from enum import Enum

import pandas as pd

from paths import TESTDATA_DIR
from app.matching.matcher import get_matches
from app.similarity.schema.embedding import embed, mean_decomp
from app.similarity.aggregation import combine_sims_weighted
from app.similarity.schema.embedding import cosine
from app.utils.input import read_headers, read_file
from app.similarity.schema.string import lev_similarity, jaccard_sim
#from app.similarity.schema.type import find_equal_types
from app.similarity.instance.numerical import find_overlap
from app.evaluation.evaluation import run_all_matchers

"""Currently mainly used for trialing and execution."""

diabetes1 = TESTDATA_DIR / 'final_datasets' / 'diabetes' /'diabetes_lv3_A.csv'
diabetes2 = TESTDATA_DIR / 'final_datasets' / 'diabetes' /'diabetes_lv3_B.csv'
gt_diabetes = TESTDATA_DIR / 'ground_truth' / 'diabetes_3_mapping.csv'

gym1 = TESTDATA_DIR / 'final_datasets' / 'gym_members' /'gym_lv3_A.csv'
gym2 = TESTDATA_DIR / 'final_datasets' / 'gym_members' /'gym_lv3_B.csv'
gt_gym = TESTDATA_DIR / 'ground_truth' / 'gym_3_mapping.csv'

result = run_all_matchers(gym1, gym2, gt_gym, ',', True, False)
print(result)
result2 = run_all_matchers(gym1, gym2, gt_gym, ',', True, True)
print(result2)

# CURRENT PROBLEM: Type checks sind nicht nuanciert genug -> führen zu hoher Varianz -> werden zu hoch gewichtet. Das ist schlecht.