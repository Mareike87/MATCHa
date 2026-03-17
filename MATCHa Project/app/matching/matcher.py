import pandas as pd
import numpy as np

def get_matches(attr1, attr2, sim_matrix, threshold):
    """Takes a similarity matrix, a threshold and two lists of attributes to be compared.
    Returns a list of tuples where each tuple represents a matched attribute and its similarity.
    The tuples are sorted by descending similarity.
    """
    if len(sim_matrix) == 0:
        return []
    sim_matrix = np.asarray(sim_matrix)
    row_index, col_index = np.where(sim_matrix >= threshold)
    attr_matches = []
    for i, j in zip(row_index, col_index):
        attr_matches.append((attr1[i], attr2[j], sim_matrix[i][j]))
    attr_matches.sort(key=lambda x: x[2], reverse=True)
    return attr_matches
