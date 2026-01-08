import numpy as np

from app.core.embeddings.embedding import get_sim


def st_sim(embedding1, embedding2):
    return get_sim(embedding1, embedding2)


def cosine(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

### PLAN:
# - take similarity matrices (calculated in core)
# - use these matrices to make a new one to combine similarity matrices -> is this better, or a sequential approach?
# - find all entries over a certain threshold -> will 1:many matches be allowed?

def combine_sim(sim1, sim2, weights):
    if weights[0]+weights[1] != 1:
        # Default to mean
        weights = (0.5, 0.5)
    result_sim = np.zeros(sim1.shape)
    for i in range(sim1.shape[0]):
        for j in range(sim1.shape[1]):
            result_sim[i][j] = (weights[0]*2*sim1[i][j] + weights[1]*2*sim2[i][j]) / 2
    return result_sim