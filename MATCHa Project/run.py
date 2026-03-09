import pandas as pd

from app.evaluation.evaluation import run_experiment, get_dataset_files
from paths import *

diabetes1 = DIABETES_DATA_DIR / 'diabetes_lv1_'
diabetes2 = DIABETES_DATA_DIR / 'diabetes_lv2_'
diabetes3 = DIABETES_DATA_DIR / 'diabetes_lv3_'
gym1 = GYM_DATA_DIR / 'gym_lv1_'
gym2 = GYM_DATA_DIR / 'gym_lv2_'
gym3 = GYM_DATA_DIR / 'gym_lv3_'
steam1 = STEAM_DATA_DIR / 'steam_lv1_'
steam2 = STEAM_DATA_DIR / 'steam_lv2_'
steam3 = STEAM_DATA_DIR / 'steam_lv3_'

diabetes = [diabetes1, diabetes2, diabetes3]
gym = [gym1, gym2, gym3]
steam = [steam1, steam2, steam3]

datasets = [diabetes1, diabetes2, diabetes3, gym1, gym2, gym3, steam1, steam2, steam3]

levels = [1,2,3]

results = []

for dataset in datasets:
    a, b, m = get_dataset_files(dataset)
    result = run_experiment(a, b, m, ',', 0.6, True, False)
    results.append({
        'dataset': dataset.name.split("_lv")[0],
        'level': dataset.name.split("_lv")[1].split("_")[0],
        'config': "Schema",
        'precision': result.get('precision'),
        'recall': result.get('recall'),
        'f1-score': result.get('f1_score'),
    })
    result = run_experiment(a, b, m, ',', 0.6, False, True)
    results.append({
        'dataset': dataset.name.split("_lv")[0],
        'level': dataset.name.split("_lv")[1].split("_")[0],
        'config': "Instance",
        'precision': result.get('precision'),
        'recall': result.get('recall'),
        'f1-score': result.get('f1_score'),
    })
    result = run_experiment(a, b, m, ',', 0.6, True, True)
    results.append({
        'dataset': dataset.name.split("_lv")[0],
        'level': dataset.name.split("_lv")[1].split("_")[0],
        'config': "Both",
        'precision': result.get('precision'),
        'recall': result.get('recall'),
        'f1-score': result.get('f1_score'),
    })
df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR / 'results.csv', index=False)