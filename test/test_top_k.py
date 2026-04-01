import numpy as np
import pandas as pd
import pytest
from app.similarity.instance.top_k import get_top_k_entries, comp_top_k, top_k_sim


def test_get_top_k_entries_basic_categorical():
    column = ["A","a","b","b","b","c"]
    top_k, valid = get_top_k_entries(column, 2, False)
    assert valid is True
    assert list(top_k.index) == ["b","a"]
    assert top_k.iloc[0] == 3


def test_get_top_k_entries_string_normalization():
    column = [" Apple","apple","APPLE ","Banana"]
    top_k, valid = get_top_k_entries(column, 2, False)
    assert "apple" in top_k.index
    assert top_k["apple"] == 3
    assert valid is True


def test_get_top_k_entries_numeric_column():
    column = [1,1,2,2,2,3]
    top_k, valid = get_top_k_entries(column, 2, True)
    assert valid is True
    assert list(top_k.index)[0] == 2
    assert top_k.iloc[0] == 3


def test_get_top_k_entries_high_uniqueness_returns_invalid():
    column = list(range(20))
    top_k, valid = get_top_k_entries(column, 5, True)
    assert valid is False


def test_get_top_k_entries_empty_column():
    column = []
    top_k, valid = get_top_k_entries(column, 3, False)
    assert valid is False
    assert top_k.empty


def test_comp_top_k_identical_sets():
    s = pd.Series([3,2], index=["a","b"])
    sim = comp_top_k(s, s)
    assert sim == 1


def test_comp_top_k_partial_overlap():
    s1 = pd.Series([3,2], index=["a","b"])
    s2 = pd.Series([4,1], index=["b","c"])
    sim = comp_top_k(s1, s2)
    assert sim == 1/3


def test_comp_top_k_no_overlap():
    s1 = pd.Series([3,2], index=["a","b"])
    s2 = pd.Series([4,1], index=["c","d"])
    sim = comp_top_k(s1, s2)
    assert sim == 0


def test_comp_top_k_empty_union():
    s1 = pd.Series(dtype=int)
    s2 = pd.Series(dtype=int)
    sim = comp_top_k(s1, s2)
    assert sim == 0


def test_top_k_sim_basic():
    df1 = pd.DataFrame({"a":["x","x","y","y","y"],"b":["a","b","c","d","e"]})
    df2 = pd.DataFrame({"c":["x","x","y","z","z"],"d":["u","v","w","x","y"]})
    sim, mask = top_k_sim(df1, df2, 2)
    assert sim.shape == (2,2)
    assert mask.shape == (2,2)
    assert mask[0,0] == 1


def test_top_k_sim_invalid_columns_mask_zero():
    df1 = pd.DataFrame({"a":list(range(20))})
    df2 = pd.DataFrame({"b":list(range(20))})
    sim, mask = top_k_sim(df1, df2, 5)
    assert mask[0,0] == 0
    assert sim[0,0] == 0


def test_top_k_sim_identical_columns_similarity_one():
    df1 = pd.DataFrame({"a":["x","x","y","y","y"]})
    df2 = pd.DataFrame({"b":["x","x","y","y","y"]})
    sim, mask = top_k_sim(df1, df2, 2)
    assert mask[0,0] == 1
    assert sim[0,0] == pytest.approx(1.0)



