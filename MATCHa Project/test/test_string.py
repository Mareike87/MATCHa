import numpy as np

from app.similarity.schema.string import lev_similarity, jaccard_sim, jaccard_word

def test_lev_similarity_identical():
    df1 = ["book"]
    df2 = ["book"]
    sim, mask = lev_similarity(df1, df2)
    assert sim[0][0] == 1
    assert mask[0][0] == 1

def test_lev_similarity_completely_different():
    df1 = ["book"]
    df2 = ["supercalifragilistic"]
    sim, mask = lev_similarity(df1, df2)
    assert sim[0][0] == 0

def test_lev_similarity_matrix_values():
    df1 = ["book","boook"]
    df2 = ["book","supercalifragilistic"]
    sim, mask = lev_similarity(df1, df2)
    expected = np.array([[1,0],[8/9,0]])
    np.testing.assert_array_equal(sim, expected)
    assert np.all(mask == 1)

def test_lev_similarity_shape():
    df1 = ["a","b","c"]
    df2 = ["a","b"]
    sim, mask = lev_similarity(df1, df2)
    assert sim.shape == (3,2)
    assert mask.shape == (3,2)

def test_lev_similarity_empty_lists():
    sim, mask = lev_similarity([], [])
    assert sim.shape == (0,0)
    assert mask.shape == (0,0)

def test_jaccard_word_identical():
    sim = jaccard_word("book","book",3)
    assert sim == 1

def test_jaccard_word_no_overlap():
    sim = jaccard_word("book","supercalifragilistic",3)
    assert sim == 0

def test_jaccard_word_partial_overlap():
    sim = jaccard_word("book","blok",3)
    assert sim == 3/9

def test_jaccard_word_case_insensitive():
    sim = jaccard_word("Book","book",3)
    assert sim == 1

def test_jaccard_word_symmetry():
    s1 = jaccard_word("hello","world",3)
    s2 = jaccard_word("world","hello",3)
    assert s1 == s2

def test_jaccard_sim_basic_matrix():
    df1 = ["book","blok"]
    df2 = ["book","back"]
    sim, mask = jaccard_sim(df1, df2, 3)
    assert sim.shape == (2,2)
    assert mask.shape == (2,2)
    assert np.all(mask == 1)

def test_jaccard_sim_invalid_token_size_low():
    df1 = ["book"]
    df2 = ["book"]
    sim, mask = jaccard_sim(df1, df2, 1)
    assert sim[0][0] == 1

def test_jaccard_sim_invalid_token_size_high():
    df1 = ["book"]
    df2 = ["book"]
    sim, mask = jaccard_sim(df1, df2, 50)
    assert sim[0][0] == 1

def test_jaccard_sim_empty_input():
    sim, mask = jaccard_sim([], [])
    assert sim.shape == (0,0)
    assert mask.shape == (0,0)