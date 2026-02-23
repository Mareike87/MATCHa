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
from app.evaluation.evaluation import run_experiment

"""Currently mainly used for trialing and execution."""

class Matching_Config:
    matchers: list
    weights: list
    threshold: float
    jac_token: int = 3
    overlap_percentile: int = 50
    prnt_sim: bool = False
    prnt_matches: bool = True

class Matching_Data:
    attributes1: list
    attributes2: list
    data1: pd.DataFrame
    data2: pd.DataFrame

class Matcher(Enum):
    JAC = 1
    LEV = 2
    EMB = 3
    EMB_MEAN = 4
    OVERLAP = 5
    DTYPES = 6

def pipeline(data:Matching_Data, config:Matching_Config):
    sims = []
    for matcher in config.matchers:
        if matcher == Matcher.JAC:
            sims.append(jaccard_sim(data.attributes1, data.attributes2, config.jac_token))
        elif matcher == Matcher.LEV:
            sims.append(lev_similarity(data.attributes1, data.attributes2))
        elif matcher == Matcher.OVERLAP:
            sims.append(find_overlap(data.data1, data.data2, config.overlap_percentile))
        # elif matcher == Matcher.DTYPES:
            # sims.append(find_equal_types(data.data1, data.data2))
        else:
            emb1 = embed(data.attributes1)
            emb2 = embed(data.attributes2)
            if matcher == Matcher.EMB:
                sims.append(cosine(emb1, emb2))
            elif matcher == Matcher.EMB_MEAN:
                e1, e2 = mean_decomp(emb1, emb2)
                sims.append(cosine(e1, e2))
    sim_matrix = combine_sims_weighted(sims, config.weights)
    if config.prnt_sim:
        print(sim_matrix)
    matches = get_matches(data.attributes1, data.attributes2, sim_matrix, config.threshold)
    if config.prnt_matches:
        print(matches)
    return sim_matrix, matches

file_name1 = TESTDATA_DIR / 'prelim_datasets' / 'diabetes_A1.csv'
file_name2 = TESTDATA_DIR / 'prelim_datasets' / 'diabetes_B1.csv'

result = run_experiment(file_name1, file_name2, "bogus", ',')
print(result)
