import numpy as np
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer

"""Model is loaded here and embeddings are generated"""

model = SentenceTransformer("google/embeddinggemma-300m")


# Returns similarity matrix of two embeddings as implemented by
# Sentence Transformers. Default is cosine.
def get_sim(embedding1, embedding2):
    return model.similarity(embedding1, embedding2)


# Returns the model currently used
def get_model():
    return model


# Returns the embeddings for a sentence or list of sentences
def embed(text):
    return model.encode(text, convert_to_numpy=True)

def mean_decomp(embedding1, embedding2):
    # embedding1 = np.array(embedding1)
    # embedding2 = np.array(embedding2)
    mean_vec = np.mean(np.vstack([embedding1, embedding2]), axis=0)
    emb1_new = embedding1 - mean_vec
    emb2_new = embedding2 - mean_vec
    emb1_new /= np.linalg.norm(emb1_new, axis=1, keepdims=True)
    emb2_new /= np.linalg.norm(emb2_new, axis=1, keepdims=True)
    return emb1_new, emb2_new

# FUNKTIONIERT NICHT
# TODO: implement mean-centering for correct pca removal
def pca_decomposition(embeddings1, embeddings2, components):
    # embeddings zusammenfassen
    embeddings = np.concatenate([embeddings1, embeddings2])
    # auf embeddings pca-decomposen
    pca = PCA(n_components=components).fit(embeddings)
    comp = pca.components_
    # ausgebesserte embeddings zurückgeben
    emb1_new = embeddings1 - (embeddings1 @ comp.T) @ comp
    emb2_new = embeddings2 - (embeddings2 @ comp.T) @ comp
    emb1_new /= np.linalg.norm(emb1_new, axis=1, keepdims=True)
    emb2_new /= np.linalg.norm(emb2_new, axis=1, keepdims=True)
    return emb1_new, emb2_new