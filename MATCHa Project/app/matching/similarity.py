import numpy as np

from app.core.embeddings.embedding import get_sim

"""Similarities are calculated here."""

# Calls get_sim from embedding.py, which calls method by SentenceTransformers
def st_sim(embedding1, embedding2):
    return get_sim(embedding1, embedding2)


# Calculates a basic cosine similarity for single vectors
def cosine(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def combine_sims(sim_matrices, weights):
    sim_matrices = np.asarray(sim_matrices)
    weights = np.asarray(weights)
    if not np.isclose(sum(weights), 1):
        # Default to equal weights
        # TODO: Add exception handling here
        print("Since weights do not sum up to 1 equal weights will be used by default.")
        weights = [1/sim_matrices.shape[0]]*sim_matrices.shape[0]
    weighted = sim_matrices * weights[:, None, None]
    result_sim = weighted.sum(axis=0)
    return result_sim



# Combines two similarity matrices using the given weights
def combine_two(sim1, sim2, weights):
    if weights[0]+weights[1] != 1:
        # Default to mean
        weights = (0.5, 0.5)
    result_sim = np.zeros(sim1.shape)
    for i in range(sim1.shape[0]):
        for j in range(sim1.shape[1]):
            result_sim[i][j] = (weights[0]*2*sim1[i][j] + weights[1]*2*sim2[i][j]) / 2
    return result_sim