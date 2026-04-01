from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESOURCES_DIR = PROJECT_ROOT / 'resources'
TESTDATA_DIR = RESOURCES_DIR / 'data'
RESULTS_DIR = RESOURCES_DIR / 'results'

DIABETES_DATA_DIR = TESTDATA_DIR / 'final_datasets' / 'diabetes'
GYM_DATA_DIR = TESTDATA_DIR / 'final_datasets' / 'gym_members'
STEAM_DATA_DIR = TESTDATA_DIR / 'final_datasets' / 'steam'
GROUND_TRUTH_DIR = TESTDATA_DIR / 'final_datasets' / 'ground_truth'