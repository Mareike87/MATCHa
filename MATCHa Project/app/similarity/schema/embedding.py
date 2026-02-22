import numpy as np
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer

"""Model is loaded here and embeddings are generated"""

model = SentenceTransformer("google/embeddinggemma-300m")

# Returns the model currently used
def get_model():
    """:return: the current sentence-transformer model"""
    return model

# Returns the embeddings for a sentence or list of sentences
def embed(text):
    """
    :param text: text to be embedded
    :return: embeddings as encoded by the used model
    """
    return model.encode(text, convert_to_numpy=True)

def mean_decomp(embedding1, embedding2):
    """
    Calculates the mean of all given embeddings and subtracts it from all the given embeddings
    :param embedding1: First set of embeddings
    :param embedding2: Second set of embeddings
    :return: The two modified sets of embeddings
    """
    mean_vec = np.mean(np.vstack([embedding1, embedding2]), axis=0)
    emb1_new = embedding1 - mean_vec
    emb2_new = embedding2 - mean_vec
    emb1_new /= np.linalg.norm(emb1_new, axis=1, keepdims=True)
    emb2_new /= np.linalg.norm(emb2_new, axis=1, keepdims=True)
    return emb1_new, emb2_new

# Calculates a basic cosine similarity for single vectors
def cosine(emb1, emb2, eps=1e-8):
    # L2-normalize
    emb1_norm = emb1 / np.maximum(np.linalg.norm(emb1, axis=1, keepdims=True), eps)
    emb2_norm = emb2 / np.maximum(np.linalg.norm(emb2, axis=1, keepdims=True), eps)

    # Cosine similarity = dot product of normalized vectors
    return (emb1_norm @ emb2_norm.T + 1.0) / 2.0, np.ones((emb1.shape[0], emb2.shape[0]))
