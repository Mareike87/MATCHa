import numpy as np
import pandas as pd

# returns upper and lower bounds of the 'middle' percentiles of size percentile
def numeric_profile(values, percentile):
    values = np.array(values)
    lower = np.percentile(values, 50-percentile/2)
    upper = np.percentile(values, 50+percentile/2)
    return lower, upper

# calculates a similarity value from the overlap of two numeric profiles
def calculate_overlap_sim(entry1, entry2):
    union = max(entry1[1][1], entry2[1][1]) - min(entry1[1][0], entry2[1][0])
    if union == 0:
        return 0
    lower = max(entry1[1][0], entry2[1][0])
    upper = min(entry1[1][1], entry2[1][1])
    overlap = 0
    if upper-lower > 0:
        overlap = upper-lower
    return overlap/union

# calls numeric_profiles and calculate_overlap_sim on all numeric columns
# returns similarity matrix and mask
def find_overlap(df1, df2, percentile):
    # get all numeric columns
    col_index_1 = [i for i, dtype in enumerate(df1.dtypes)
                   if pd.api.types.is_numeric_dtype(dtype)]
    col_index_2 = [i for i, dtype in enumerate(df2.dtypes)
                   if pd.api.types.is_numeric_dtype(dtype)]
    # save numeric profiles of all numeric columns
    profiles_1 = []
    for idx in col_index_1:
        profiles_1.append((idx, numeric_profile(df1.iloc[:, idx], percentile)))
    profiles_2 = []
    for idx in col_index_2:
        profiles_2.append((idx, numeric_profile(df2.iloc[:, idx], percentile)))
    # initialize sim_matrix and mask
    sim_matrix = np.zeros((df1.shape[1], df2.shape[1]))
    mask = np.zeros((df1.shape[1], df2.shape[1]))
    # get similarity of each pair of profiles
    for entry1 in profiles_1:
        for entry2 in profiles_2:
            sim_matrix[entry1[0], entry2[0]] = calculate_overlap_sim(entry1, entry2)
            mask[entry1[0], entry2[0]] = 1
    return sim_matrix, mask