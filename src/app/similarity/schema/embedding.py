import numpy as np

from sentence_transformers import SentenceTransformer

model = None

def load_model():
    global model
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("google/embeddinggemma-300m")
        except Exception as e:
            print("Embedding model could not be loaded:", e)
            model = None
    return model

# Returns the model currently used
def get_model():
    return model

# Returns the embeddings for a sentence or list of sentences
def embed(text):
    mdl = load_model()
    if mdl is None:
        raise RuntimeError("Embedding model not available")
    return mdl.encode(text, convert_to_numpy=True)

# subtracts the shared mean vector of embedding1 and embedding two from all vectors
def mean_decomp(embedding1, embedding2):
    mean_vec = np.mean(np.vstack([embedding1, embedding2]), axis=0)
    emb1_new = embedding1 - mean_vec
    emb2_new = embedding2 - mean_vec
    return emb1_new, emb2_new

# Calculates a basic cosine similarity for single vectors
def cosine(emb1, emb2, eps=1e-8):
    # L2-normalize
    emb1_norm = emb1 / np.maximum(np.linalg.norm(emb1, axis=1, keepdims=True), eps)
    emb2_norm = emb2 / np.maximum(np.linalg.norm(emb2, axis=1, keepdims=True), eps)
    # Cosine similarity = dot product of normalized vectors
    return (emb1_norm @ emb2_norm.T + 1.0) / 2.0, np.ones((emb1.shape[0], emb2.shape[0]))
