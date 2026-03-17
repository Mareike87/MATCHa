import time
import pandas as pd
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
                    'precision': result.get('precision'),
                    'recall': result.get('recall'),
                    'f1-score': result.get('f1_score'),
                })
                result = run_experiment(a, b, m, ',', 0.6, False, True)
                results.append({
                    'dataset': dataset[i][1],
                    'level': i+1,
                    'config': "Instance",
                    'precision': result.get('precision'),
                    'recall': result.get('recall'),
                    'f1-score': result.get('f1_score'),
                })
                result = run_experiment(a, b, m, ',', 0.6, True, True)
                results.append({
                    'dataset': dataset[i][1],
                    'level': i+1,
                    'config': "Both",
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

def aggregate_metrics(df):
    agg = (
        df.groupby(["dataset", "level", "config"])
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
        )
        .reset_index()
    )
    return agg

def plot_lineplots(df, metric="f1"):
    sns.set_theme(style="whitegrid")
    datasets = df["dataset"].unique()
    for dataset in datasets:
        plt.figure()
        subset = df[df["dataset"] == dataset]
        sns.lineplot(
            data=subset,
            x="level",
            y=metric,
            hue="config",
            estimator="mean",
            errorbar="sd",
            marker="o"
        )
        plt.title(f"{dataset} - {metric.upper()} over Difficulty")
        plt.xlabel("Difficulty Level")
        plt.ylabel(metric.upper())
        plt.legend(title="Config")
        plt.tight_layout()
        plt.show()

def plot_boxplots(df, metric="f1"):
    sns.set_theme(style="whitegrid")
    datasets = df["dataset"].unique()
    for dataset in datasets:
        plt.figure()
        subset = df[df["dataset"] == dataset]
        sns.boxplot(
            data=subset,
            x="level",
            y=metric,
            hue="config"
        )
        plt.title(f"{dataset} - {metric.upper()} Distribution")
        plt.xlabel("Difficulty Level")
        plt.ylabel(metric.upper())

        plt.legend(title="Config")
        plt.tight_layout()
        plt.show()

def plot_precision_recall_scatter(df):
    sns.set_theme(style="whitegrid")
    agg = (
        df.groupby(["dataset", "level", "config"])
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean")
        )
        .reset_index()
    )
    plt.figure()
    sns.scatterplot(
        data=agg,
        x="recall",
        y="precision",
        hue="config",
        style="dataset",
        size="level",
        sizes=(50, 150)
    )
    plt.title("Precision vs Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

df = load_data(RESULTS_DIR / "results.csv")

# Linienplots (Fokus auf F1)
plot_lineplots(df, metric="f1")

# Optional auch für precision/recall
plot_lineplots(df, metric="precision")
plot_lineplots(df, metric="recall")

# Boxplots (Streuung)
plot_boxplots(df, metric="f1")

# Precision-Recall Tradeoff
plot_precision_recall_scatter(df)