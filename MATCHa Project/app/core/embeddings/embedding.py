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
    return model.encode(text)