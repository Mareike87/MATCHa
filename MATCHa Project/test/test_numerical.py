import numpy as np
import pandas as pd
import pytest
from app.similarity.instance.numerical import numeric_profile, calculate_overlap_sim, find_overlap


def test_numeric_profile_basic():
    values = [1, 2, 3, 4, 5]
    lower, upper = numeric_profile(values, 50)
    assert lower == np.percentile(values, 25)
    assert upper == np.percentile(values, 75)


def test_numeric_profile_single_value():
    values = [5, 5, 5]
    lower, upper = numeric_profile(values, 50)
    assert lower == 5
    assert upper == 5


def test_numeric_profile_percentile_zero():
    values = [1, 2, 3, 4]
    lower, upper = numeric_profile(values, 0)
    assert lower == np.percentile(values, 50)
    assert upper == np.percentile(values, 50)


def test_calculate_overlap_sim_full_overlap():
    entry1 = (0, (0, 10))
    entry2 = (1, (0, 10))
    sim = calculate_overlap_sim(entry1, entry2)
    assert sim == 1


def test_calculate_overlap_sim_partial_overlap():
    entry1 = (0, (0, 10))
    entry2 = (1, (5, 15))
    sim = calculate_overlap_sim(entry1, entry2)
    assert sim == 5/15


def test_calculate_overlap_sim_no_overlap():
    entry1 = (0, (0, 5))
    entry2 = (1, (10, 15))
    sim = calculate_overlap_sim(entry1, entry2)
    assert sim == 0


def test_calculate_overlap_sim_zero_union():
    entry1 = (0, (5, 5))
    entry2 = (1, (5, 5))
    sim = calculate_overlap_sim(entry1, entry2)
    assert sim == 0


def test_find_overlap_basic_numeric_columns():
    df1 = pd.DataFrame({"a":[1,2,3,4],"b":[10,11,12,13]})
    df2 = pd.DataFrame({"x":[2,3,4,5],"y":[20,21,22,23]})
    sim_matrix, mask = find_overlap(df1, df2, 50)
    assert sim_matrix.shape == (2,2)
    assert mask.shape == (2,2)
    assert np.all(mask == 1)


def test_find_overlap_with_non_numeric_columns():
    df1 = pd.DataFrame({"a":[1,2,3],"b":["x","y","z"]})
    df2 = pd.DataFrame({"c":[1,2,3],"d":["a","b","c"]})
    sim_matrix, mask = find_overlap(df1, df2, 50)
    assert sim_matrix.shape == (2,2)
    assert mask[0,0] == 1
    assert mask[1,0] == 0
    assert mask[0,1] == 0
    assert mask[1,1] == 0


def test_find_overlap_no_numeric_columns():
    df1 = pd.DataFrame({"a":["x","y"],"b":["z","w"]})
    df2 = pd.DataFrame({"c":["a","b"],"d":["c","d"]})
    sim_matrix, mask = find_overlap(df1, df2, 50)
    assert np.all(sim_matrix == 0)
    assert np.all(mask == 0)


def test_find_overlap_identical_numeric_columns_high_similarity():
    df1 = pd.DataFrame({"a":[1,2,3,4,5]})
    df2 = pd.DataFrame({"b":[1,2,3,4,5]})
    sim_matrix, mask = find_overlap(df1, df2, 50)
    assert mask[0,0] == 1
    assert sim_matrix[0,0] == pytest.approx(1.0)