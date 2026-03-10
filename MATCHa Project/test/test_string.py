from app.similarity.schema.string import *

# Tests for Levenshtein
def test_lev_similarity1():
    df1 = ["book"]
    df2 = ["book"]
    sim, mask = lev_similarity(df1, df2)
    assert sim[0][0] == 1

def test_lev_similarity2():
    df1 = ["book"]
    df2 = ["supercalifragilistic"]
    sim, mask = lev_similarity(df1, df2)
    print(sim[0][0])
    assert sim[0][0] == 0

def test_lev_similarity3():
    df1 = ["book", "boook"]
    df2 = ["book", "supercalifragilistic"]
    sim, mask = lev_similarity(df1, df2)
    result = np.array([[1,0],[8/9,0]])
    np.testing.assert_array_equal(sim, result)

def test_jaccard_word1():
    sim = jaccard_word("book", "book", 3)
    assert sim == 1

def test_jaccard_word2():
    sim = jaccard_word("book", "supercalifragilistic", 3)
    assert sim == 0

def test_jaccard_word3():
    sim = jaccard_word("book", "blok", 3)
    result = 3/9
    assert sim == result
