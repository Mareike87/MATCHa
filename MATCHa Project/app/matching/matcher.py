import pandas as pd
import numpy as np

def get_matches(attr1, attr2, sim_matrix, threshold):
    """Takes a similarity matrix, a threshold and two lists of attributes to be compared.
    Returns a list of tuples where each tuple represents a matched attribute and its similarity.
    The tuples are sorted by descending similarity.
    """
    #Annahme: sim_matrix ist ein np Array
    over_threshold = np.nonzero(sim_matrix >= threshold)
    attr_matches = []
    for i in range(over_threshold[0].shape[0]):
        attr_matches.append((attr1[over_threshold[0][i]], attr2[over_threshold[1][i]], sim_matrix[over_threshold[0][i], over_threshold[1][i]]))
    attr_matches.sort(key=lambda x: x[2], reverse=True)
    return attr_matches


## TODO:
def get_top_k_matches(attr1, attr2, sim_matrix, k):
    sim_matrix.nlargest(k, "")