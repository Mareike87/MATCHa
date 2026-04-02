import pandas as pd
import numpy as np

# Takes a similarity matrix, a threshold and two lists of attributes to be compared.
# Returns a list of triples where each triple represents two matched attributes and their similarity.
# The triples are sorted by descending similarity.
def get_matches(attr1, attr2, sim_matrix, threshold):
    if len(sim_matrix) == 0:
        return []
    sim_matrix = np.asarray(sim_matrix)
    row_index, col_index = np.where(sim_matrix >= threshold)
    attr_matches = []
    for i, j in zip(row_index, col_index):
        attr_matches.append((attr1[i], attr2[j], sim_matrix[i][j]))
    attr_matches.sort(key=lambda x: x[2], reverse=True)
    return attr_matches
