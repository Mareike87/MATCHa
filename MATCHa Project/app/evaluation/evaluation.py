import time

from app.utils.input import read_headers, read_file
from app.similarity.schema.embedding import embed, cosine
from app.similarity.aggregation import combine_sims_var
from app.matching.matcher import get_matches

# CAUTION: this is temporary *only*
gt = [("Pregnancies", "Pregn;Glucose"),
      ("Glucose", "Pregn;Glucose"),
      ("BloodPressure", "BloodPressure"),
      ("SkinThickness", "6i2_SkinThickness"),
      ("Insulin", "Insulin"),
      ("BMI", "BMI"),
      ("DiabetesPedigreeFunction", "DiabetesPedigreeFunction"),
      ("Age", "Age"),
      ("Outcome", "Outc")]

def run_experiment(datapath1, datapath2, matcher, delimiter):
    # TODO: organized way to communicate which matcher should be used and delimiters
    start = time.time()
    headers1 = read_headers(datapath1, delimiter)
    headers2 = read_headers(datapath2, delimiter)
    emb1 = embed(headers1)
    emb2 = embed(headers2)
    sim_embed, mask = cosine(emb1, emb2)
    sim_final = combine_sims_var(sim_embed, mask)
    matches = get_matches(headers1, headers2, sim_final, 0.9)
    matches = [m[:2] for m in matches]
    end = time.time()
    # Assumption: we have a ground truth called gt which contains a list of matches in the format of (attr_name1, attr_name2)
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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    runtime = end - start
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "runtime": runtime
    }
