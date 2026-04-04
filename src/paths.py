from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESOURCES_DIR = PROJECT_ROOT / 'resources'
TESTDATA_DIR = RESOURCES_DIR / 'data'
RESULTS_DIR = RESOURCES_DIR / 'results'

DIABETES_DATA_DIR = TESTDATA_DIR / 'example_datasets' / 'diabetes'
GYM_DATA_DIR = TESTDATA_DIR / 'example_datasets' / 'gym_members'
STEAM_DATA_DIR = TESTDATA_DIR / 'example_datasets' / 'steam'
DEMO_PATH = RESOURCES_DIR / 'demo_files'