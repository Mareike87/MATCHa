import time

from app.pipeline.matching_manager import run_matching
from app.similarity.instance.numerical import find_overlap
from app.similarity.instance.top_k import top_k_sim
from app.similarity.schema.string import jaccard_sim, lev_similarity
from app.similarity.schema.type import find_type_similarity
from app.utils.input import read_headers, read_file, read_mappings
from app.similarity.schema.embedding import embed, cosine, mean_decomp
from app.similarity.aggregation import combine_sims_var
from app.matching.matcher import get_matches
from paths import TESTDATA_DIR


def get_dataset_files(base):
    base = str(base)
    return (
        base + "A.csv",
        base + "B.csv",
        base + "map.csv"
    )

def run_experiment(datapath1, datapath2, gt_file, delimiter, threshold, schema, instance):
    gt = read_mappings(gt_file)
    start = time.time()
    matches = run_matching(datapath1, datapath2, delimiter, threshold, schema, instance, )
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
        precision = None
    recall = tp / (tp + fn)
    if precision is None or recall+precision == 0:
        f1_score = None
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    runtime = end - start
    #print(matches)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "runtime": runtime
    }


