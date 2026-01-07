from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")


def get_sim(embedding1, embedding2):
    return model.similarity(embedding1, embedding2)


def get_model():
    return model


def embed(text):
    return model.encode(text)