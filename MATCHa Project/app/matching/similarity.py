import numpy as np

from app.core.embeddings.embedding import get_sim

"""Similarities are calculated here."""

# Calls get_sim from embedding.py, which calls method by SentenceTransformers
def st_sim(embedding1, embedding2):
    return get_sim(embedding1, embedding2)


# Calculates a basic cosine similarity for single vectors
def cosine(emb1, emb2, eps=1e-8):
    # L2-normalize
    emb1_norm = emb1 / np.maximum(np.linalg.norm(emb1, axis=1, keepdims=True), eps)
    emb2_norm = emb2 / np.maximum(np.linalg.norm(emb2, axis=1, keepdims=True), eps)

    # Cosine similarity = dot product of normalized vectors
    return (emb1_norm @ emb2_norm.T + 1.0) / 2.0

def combine_sims(sim_matrices, weights=None, clip=True):
    sim_matrices = np.asarray(sim_matrices)
    # If only one matrix is given transform shape to allow computation
    if sim_matrices.ndim == 2:
        sim_matrices = sim_matrices[None, :, :]
    # If no weights are given set equal weights
    if weights is None:
        weights = np.full(sim_matrices.shape[0], 1/sim_matrices.shape[0])
    else:
        weights = np.asarray(weights)
        # If weights do not sum to 1 set equal weights
        if not np.isclose(sum(weights), 1):
            # TODO: Add exception handling here
            print("Since weights do not sum up to 1 equal weights will be used by default.")
            weights = np.full(sim_matrices.shape[0], 1/sim_matrices.shape[0])

    weighted = sim_matrices * weights[:, None, None]
    result_sim = weighted.sum(axis=0)
    # Clip to [0,1] since float calculations may lead to values outside this range
    if clip:
        result_sim = np.clip(result_sim, 0, 1)

    return result_sim



# Combines two similarity matrices using the given weights
def combine_two(sim1, sim2, weights):
    if weights[0]+weights[1] != 1:
        # Default to mean
        weights = (0.5, 0.5)
    result_sim = np.zeros(sim1.shape)
    for i in range(sim1.shape[0]):
        for j in range(sim1.shape[1]):
            result_sim[i][j] = weights[0]*sim1[i][j] + weights[1]*sim2[i][j]
    return result_sim