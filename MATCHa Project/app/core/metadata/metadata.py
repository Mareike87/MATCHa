import pandas as pd
import numpy as np

## NOTE TO SELF: shape[0] = Zeilen, shape[1] = Spalten
def find_equal_types(df1, df2):
    #.dtypes = Series mit index Attributname und value Type
    #.values =
    types1 = df1.dtypes.values[:, None]
    types2 = df2.dtypes.values[None, :]

    return (types1 == types2).astype(int)

def numeric_profile(values, percentile):
    values = np.array(values)
    lower = np.percentile(values, 50-percentile/2)
    upper = np.percentile(values, 50+percentile/2)
    return lower, upper

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

# should:
# 1. take two datasets and a percentile
# 2. take all columns with numeric values
# 3. call numeric_profile on them to find lower and upper values surrounding the middle percentage
# 4. call calculate_overlap on each pair of column values to find value overlap
# 5. return matrix containing the overlap results
# -> problem 1: does the matrix only contain numeric cols or all filled with 0s?
#     -> 0s difficult as they mean "no overlap" may skew later results
# -> problem 2: how to normalize the overlap results?
#     -> which percentage of each percentile is covered by both?
#       -> leads to another issue: whose percentage counts? a may cover 100% of b, but b only 50% of a

# Lösung: intersection/union eintragen -> find source for this approach

def find_overlap(df1, df2, percentile):
    col_index_1 = [i for i, dtype in enumerate(df1.dtypes)
                   if pd.api.types.is_numeric_dtype(dtype)]
    col_index_2 = [i for i, dtype in enumerate(df2.dtypes)
                   if pd.api.types.is_numeric_dtype(dtype)]
    profiles_1 = []
    for idx in col_index_1:
        profiles_1.append((idx, numeric_profile(df1.iloc[:, idx], percentile)))
    profiles_2 = []
    for idx in col_index_2:
        profiles_2.append((idx, numeric_profile(df2.iloc[:, idx], percentile)))

    sim_matrix = np.full((df1.shape[1], df2.shape[1]), 0.5)

    for entry1 in profiles_1:
        for entry2 in profiles_2:
            sim_matrix[entry1[0], entry2[0]] = calculate_overlap_sim(entry1, entry2)
    return sim_matrix