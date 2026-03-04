import time
from wsgiref import headers

from app.similarity.instance.numerical import find_overlap
from app.similarity.instance.top_k import top_k_sim
from app.similarity.schema.string import jaccard_sim, lev_similarity
from app.similarity.schema.type import find_type_similarity
from app.utils.input import read_headers, read_file, read_mappings
from app.similarity.schema.embedding import embed, cosine, mean_decomp
from app.similarity.aggregation import combine_sims_var
from app.matching.matcher import get_matches


def run_all_matchers(datapath1, datapath2, gt_file, delimiter, schema, instance):
    gt = read_mappings(gt_file)
    start = time.time()
    similarities = []
    masks = []
    if schema: # caution: sollte hier nur embeddings || mean embed?
        headers1 = read_headers(datapath1, delimiter)
        headers2 = read_headers(datapath2, delimiter)
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
        sim, mask = find_overlap(df1, df2, 50)      # 50 or different? adjustment opportunity
        similarities.append(sim)
        masks.append(mask)
        # top k
        sim, mask = top_k_sim(df1, df2, 50)         # 50 or different? adjustment opportunity
        similarities.append(sim)
        masks.append(mask)
    sim_final = combine_sims_var(similarities, masks)
    matches = get_matches(headers1, headers2, sim_final, 0.9)
    matches = [m[:2] for m in matches]
    end = time.time()
    tp = 0
    fp = 0
    fn = 0
    for match in gt:
        if match in matches: # note: comparison mus be done differently, esp to insure order of attributes does not matter
            tp += 1
        else:
            fn += 1
    for match in matches:
        if not match in gt:
            fp += 1
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        print("sorry, your matcher is so shit it hasn't found a single match")
        return
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        print("your matcher appears to have somehow found nothing and everything at the same time. How is this possible?")
    f1_score = 2 * precision * recall / (precision + recall)
    runtime = end - start
    print(matches)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "runtime": runtime
    }
