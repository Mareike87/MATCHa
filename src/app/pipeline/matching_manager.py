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

def run_matching(datapath1, datapath2, delimiter, threshold,schema=True, instance=True):
    similarities = []
    masks = []
    headers1 = read_headers(datapath1, delimiter)
    headers2 = read_headers(datapath2, delimiter)
    if not schema and not instance:
        print("please select either schema, instance or both")
        return None
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
