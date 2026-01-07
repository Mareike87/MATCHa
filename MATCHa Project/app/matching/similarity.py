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

