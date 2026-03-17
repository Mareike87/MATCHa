import time
import pandas as pd
import matplotlib.pyplot as plt

from app.pipeline.matching_manager import run_matching
from app.utils.input import read_mappings
from paths import *

def get_dataset_files(base):
    base = str(base)
    return (
        base + "A1.csv",
        base + "B1.csv",
        base + "Mapping.csv"
    )

def run_experiment(datapath1, datapath2, gt_file, delimiter, threshold, schema, instance):
    gt = read_mappings(gt_file)
    start = time.time()
    matches = run_matching(datapath1, datapath2, delimiter, threshold, schema, instance)
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

def make_graph(csv_path):
    # CSV einlesen
    df = pd.read_csv(csv_path)

    # Nach Level gruppieren und Mittelwerte berechnen
    grouped = df.groupby("config")[["precision", "recall", "f1-score"]].mean().reset_index()

    # Sortieren nach Level (1,2,3)
    grouped = grouped.sort_values("config")

    # Plot erstellen
    plt.figure()

    plt.plot(grouped["config"], grouped["precision"], marker="o", label="Precision")
    plt.plot(grouped["config"], grouped["recall"], marker="o", label="Recall")
    plt.plot(grouped["config"], grouped["f1-score"], marker="o", label="F1-score")


    plt.xlabel("Config")
    plt.ylabel("Score")
    plt.title("Precision, Recall und F1-Score nach Config")
    plt.xticks([1, 2, 3])
    plt.legend()
    plt.grid(True)

    plt.show()

path = RESULTS_DIR / 'results.csv'
make_graph(path)
