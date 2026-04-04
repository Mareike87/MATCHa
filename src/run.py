from app.evaluation.evaluation import get_dataset_files
from app.pipeline.matching_manager import run_matching
from app.utils.input import read_mappings
from paths import *


# this file runs a basic demo of MATCHa, excluding embeddings
# if you want to include embeddings, make sure you have a valid access token for HuggingFace and get access
# to embedding gemma (or choose a different model in embedding.py)
# then you can set includeEmbeddings to 'True' below

# for more testing datasets see the example_datasets folder.
# the paths for the directories 'diabetes', 'gym_members' and 'steam' are available in paths.py
# in order to use one of these use the following statement (replace X with the complexity level of your choice)
# demo_files = DIABETES_DATA_DIR / 'diabetes_lvX_'

demo_files = DEMO_PATH / 'demo_'

results = []

a, b, m = get_dataset_files(demo_files)

schema = input("Would you like to use schema-based matchers? (y/n): ")
if schema == 'y': schema = True
else: schema = False

instance = input("Would you like to use instance-based matchers? (y/n): ")
if instance == 'y': instance = True
else: instance = False

if not instance and not schema:
    print("You did not specify a matching configuration! Combined configuration will be used by default.")
    instance = True
    schema = True

check = True
while check:
    threshold = input("Which threshold do you want to use? (thresholds between 0.4 and 0.9 are recommended):")
    try:
        threshold = float(threshold)
        check = False
    except ValueError:
        print("Invalid threshold value. Please try again.")

matches = run_matching(a, b, ',', threshold, schema, instance, False)
matches = [(x,y) for x,y,z in matches]
matches.sort()

demo_gt = read_mappings(m)
demo_gt = list(demo_gt)
demo_gt.sort()

print("The following matches were found by the matching algorithm:")
print(matches)
print("The following matches are in the ground truth:")
print(demo_gt)