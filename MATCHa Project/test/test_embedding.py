import numpy as np
import pytest

from sentence_transformers import SentenceTransformer

from app.similarity.schema.embedding import get_model, embed, mean_decomp, cosine


def test_get_model_returns_sentence_transformer():
    model = get_model()
    assert isinstance(model, SentenceTransformer)


def test_embed_single_string_returns_numpy():
    emb = embed("hello world")
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1 or emb.ndim == 2


def test_embed_list_returns_multiple_embeddings():
    texts = ["hello world","another sentence"]
    emb = embed(texts)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == 2


def test_mean_decomp_shapes_preserved():
    emb1 = np.random.rand(3,5)
    emb2 = np.random.rand(4,5)
    e1, e2 = mean_decomp(emb1, emb2)
    assert e1.shape == emb1.shape
    assert e2.shape == emb2.shape


def test_mean_decomp_normalized_vectors():
    emb1 = np.random.rand(2,4)
    emb2 = np.random.rand(2,4)
    e1, e2 = mean_decomp(emb1, emb2)
    norms1 = np.linalg.norm(e1, axis=1)
    norms2 = np.linalg.norm(e2, axis=1)
    assert np.allclose(norms1, 1)
    assert np.allclose(norms2, 1)


def test_mean_decomp_mean_centering():
    emb1 = np.array([[1.0,2.0],[3.0,4.0]])
    emb2 = np.array([[5.0,6.0]])
    e1, e2 = mean_decomp(emb1, emb2)
    combined = np.vstack([e1,e2])
    assert combined.shape[1] == 2


def test_cosine_identical_vectors():
    v = np.array([[1.0,0.0]])
    sim, mask = cosine(v, v)
    assert sim.shape == (1,1)
    assert mask.shape == (1,1)
    assert sim[0,0] == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    v1 = np.array([[1.0,0.0]])
    v2 = np.array([[0.0,1.0]])
    sim, mask = cosine(v1, v2)
    assert sim[0,0] == pytest.approx(0.5)


def test_cosine_matrix_shape():
    emb1 = np.random.rand(3,5)
    emb2 = np.random.rand(4,5)
    sim, mask = cosine(emb1, emb2)
    assert sim.shape == (3,4)
    assert mask.shape == (3,4)
    assert np.all(mask == 1)


def test_cosine_handles_zero_vector():
    emb1 = np.array([[0.0,0.0]])
    emb2 = np.array([[1.0,0.0]])
    sim, mask = cosine(emb1, emb2)
    assert sim.shape == (1,1)