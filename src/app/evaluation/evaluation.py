import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from app.pipeline.matching_manager import run_matching
from app.utils.input import read_mappings
from paths import *

diabetes1path = 'C:/Users/Steff/Desktop/Generated Datasets/Diabetes/Diabetes_Level_1/' + 'diab_lv1_'
diabetes2path = 'C:/Users/Steff/Desktop/Generated Datasets/Diabetes/Diabetes_Level_2/' + 'diab_lv2_'
diabetes3path = 'C:/Users/Steff/Desktop/Generated Datasets/Diabetes/Diabetes_Level_3/' + 'diab_lv3_'
gym1path = 'C:/Users/Steff/Desktop/Generated Datasets/Gym/Gym_Level_1/' + 'gym_lv1_'
gym2path = 'C:/Users/Steff/Desktop/Generated Datasets/Gym/Gym_Level_2/' + 'gym_lv2_'
gym3path = 'C:/Users/Steff/Desktop/Generated Datasets/Gym/Gym_Level_3/' + 'gym_lv3_'
steam1path = 'C:/Users/Steff/Desktop/Generated Datasets/Steam/Steam_Level_1/' + 'steam_lv1_'
steam2path = 'C:/Users/Steff/Desktop/Generated Datasets/Steam/Steam_Level_2/' + 'steam_lv2_'
steam3path = 'C:/Users/Steff/Desktop/Generated Datasets/Steam/Steam_Level_3/' + 'steam_lv3_'

diabetes = [(diabetes1path, "Diabetes"), (diabetes2path, "Diabetes"), (diabetes3path, "Diabetes")]
gym = [(gym1path, "Gym"), (gym2path, "Gym"), (gym3path, "Gym")]
steam = [(steam1path, "Steam"), (steam2path, "Steam"), (steam3path, "Steam")]

datasets = [diabetes, gym, steam]

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
    sims = [m[2] for m in matches]
    mean_sim = np.mean(sims)
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
        "avg_similarity": mean_sim,
        "runtime": runtime
    }

def runeach_and_save():
    c = 0
    results = []
    for dataset in datasets:
        for i in range(0,3):
            for j in range(1, 101):
                a, b, m = get_dataset_files(dataset[i][0] + str(j) + '_')
                result = run_experiment(a, b, m, ',', 0.6, True, False)
                results.append({
                    'dataset': dataset[i][1],
                    'level': i+1,
                    'config': "Schema",
                    'avg_similarity': result.get('avg_similarity'),
                    'precision': result.get('precision'),
                    'recall': result.get('recall'),
                    'f1-score': result.get('f1_score'),
                })
                result = run_experiment(a, b, m, ',', 0.6, False, True)
                results.append({
                    'dataset': dataset[i][1],
                    'level': i+1,
                    'config': "Instance",
                    'avg_similarity': result.get('avg_similarity'),
                    'precision': result.get('precision'),
                    'recall': result.get('recall'),
                    'f1-score': result.get('f1_score'),
                })
                result = run_experiment(a, b, m, ',', 0.6, True, True)
                results.append({
                    'dataset': dataset[i][1],
                    'level': i+1,
                    'config': "Combined",
                    'avg_similarity': result.get('avg_similarity'),
                    'precision': result.get('precision'),
                    'recall': result.get('recall'),
                    'f1-score': result.get('f1_score'),
                })
                c += 1
                print(c)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'results.csv', index=False)
    return

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"f1-score": "f1"})
    return df

