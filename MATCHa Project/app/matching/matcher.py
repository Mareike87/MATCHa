
def get_matches(attr1, attr2, sim_matrix, threshold):
    """Takes a similarity matrix, a threshold and two lists of attributes to be compared.
    Returns a list of tuples where each tuple represents a matched attribute and its similarity.
    The tuples are sorted by descending similarity.
    """
    scored = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix[i].shape[0]):
            if sim_matrix[i][j] > threshold:
                scored.append((i,j, sim_matrix[i][j]))
    scored.sort(key=lambda x: x[2], reverse=True)
    attr_matches = []
    for match in scored:
        attr_matches.append((attr1[match[0]], attr2[match[1]], match[2]))
    return attr_matches
