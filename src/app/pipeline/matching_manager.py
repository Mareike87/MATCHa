from app.similarity.instance.top_k import top_k_sim
from app.similarity.schema.type import find_type_similarity
from app.matching.matcher import get_matches
from app.similarity.schema.embedding import embed, mean_decomp
from app.similarity.aggregation import combine_sims_var
from app.similarity.schema.embedding import cosine
from app.utils.input import read_headers, read_file
from app.similarity.schema.string import lev_similarity, jaccard_sim
from app.similarity.instance.numerical import find_overlap

# runs the matching pipeline based on the given parameters
def run_matching(datapath1, datapath2, delimiter, threshold,schema=True, instance=True):
    similarities = []
    masks = []
    headers1 = read_headers(datapath1, delimiter)
    headers2 = read_headers(datapath2, delimiter)
    if not schema and not instance:
        print("please select either schema, instance or both")
        return None
    if schema:
        # generate embeddings
        emb1 = embed(headers1)
        emb2 = embed(headers2)
        # cosine similarity on normal embeddings
        sim, mask = cosine(emb1, emb2)
        similarities.append(sim)
        masks.append(mask)
        # mean decomposition and cosine similarity
        emb1, emb2 = mean_decomp(emb1, emb2)
        sim, mask = cosine(emb1, emb2)
        similarities.append(sim)
        masks.append(mask)
        # Jaccard similarity
        sim, mask = jaccard_sim(headers1, headers2, 3)  # 3 or differen? adjustment opportunity
        similarities.append(sim)
        masks.append(mask)
        # Levenshtein similarity
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
        # percentile range
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
