import numpy as np
import pandas as pd

# returns the top k most frequent values of a column
def get_top_k_entries(column, k, isNumber):
    column = pd.Series(column)
    if not isNumber: # conversion to string with normalization if non-numeric values
        column = (column.astype(str)
                  .str.lower()
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True))
    num_unique_entries = column.value_counts().count()
    top_k = column.value_counts().head(k)
    # check for validity of entries
    if num_unique_entries/column.count() > 0.9 or top_k.count() == 0:
        return top_k, False
    return top_k, True

# compares two sets of top-k most frequent values
# returns their similarity
def comp_top_k(top_k1, top_k2):
    i = top_k1.index
    j = top_k2.index
    intersection = set(i).intersection(set(j))
    union = set(i).union(set(j))
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

# calls get_top_k_entries and comp_top_k on all attribute pairs
def top_k_sim(df1, df2, k):
    m = df1.shape[1]
    n = df2.shape[1]
    sim = np.zeros((m,n))
    mask = np.ones((m,n))
    top_k1 = []
    valid1 = []
    for col in df1.columns:
        top_k, valid = get_top_k_entries(df1[col], k, pd.api.types.is_numeric_dtype(df1[col]))
        top_k1.append(top_k)
        valid1.append(valid)
    top_k2 = []
    valid2 = []
    for col in df2.columns:
        top_k, valid = get_top_k_entries(df2[col], k, pd.api.types.is_numeric_dtype(df2[col]))
        top_k2.append(top_k)
        valid2.append(valid)
    # check for validity, fill sim and mask
    for i in range(m):
        for j in range(n):
            if valid1[i] and valid2[j]:
                sim[i, j] = comp_top_k(top_k1[i], top_k2[j])
            else: mask[i, j] = 0
    return sim, mask