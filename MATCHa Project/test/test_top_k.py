import pandas as pd
import numpy as np

from app.similarity.instance.top_k import *

# To test:
# Run all tests: pytest
# Run all tests with coverage: pytest --cov=app.similarity.instance.top_k
# Run all tests with coverage + report: pytest --cov=app.similarity.instance.top_k --cov-report=term-missing


# Tests for get_top_k_entries
## Missing: more potential entries than k
def test_removal_unique():
    # Tests for removal of a value that only appears once, as well as string normalizations
    col = ["got", "got", "be", "As", "as", "  as", "bring me", "bring  me"]
    result, ignore = get_top_k_entries(col, 50, False)
    expected = pd.Series({"as": 3, "got": 2, "bring me": 2}, name="count")
    pd.testing.assert_series_equal(result, expected, check_dtype=False)

def test_no_candidates():
    # Tests that no viable candidates for top_k leads to empty series
    col = ["", "got", "as", "had"]
    result, ignore = get_top_k_entries(col, 50, False)
    assert result.empty == True

# Tests for comp_top_k
def test_correct_comp():
    col1 = ["got", "got", "be", "As", "as", "  as", "bring me", "bring  me"]
    col2 = ["got", "had", "had  ", "as", "As", "bring me", "bring me", "be", "Be", "BE"]
    top_1, ignore = get_top_k_entries(col1, 50, False) # got, as, bring me
    top_2, ignore = get_top_k_entries(col2, 50, False) # had, as, bring me, be
    result = comp_top_k(top_1, top_2)
    assert result == 2/5

def test_no_intersection():
    col1 = ["got", "got", "be", "As", "as", "  as", "bring me", "bring  me"]
    col2 = ["had", "had ", "none", "blue", "BLUE"]
    top_1, ignore = get_top_k_entries(col1, 50, False)
    top_2, ignore = get_top_k_entries(col2, 50, False)
    result = comp_top_k(top_1, top_2)
    assert result == 0

def test_no_union():
    col1 = ["got", "be", "As", "bring me"]
    col2 = ["had", "haid ", "none", "blues", "BLUE"]
    top_1, ignore = get_top_k_entries(col1, 50, False)
    top_2, ignore = get_top_k_entries(col2, 50, False)
    result = comp_top_k(top_1, top_2)
    assert result == 0

def test_comp_empty():
    col1 = ["", "got", "as", "had"]
    col2 = ["had", "had ", "none", "blue", "BLUE"]
    top_1, ignore = get_top_k_entries(col1, 50, False)
    top_2, ignore = get_top_k_entries(col2, 50, False)
    result = comp_top_k(top_1, top_2)
    assert result == 0

# Tests for top_k_sim
def test_correct_sim():
    data1 = {
        "col1": ["got", "got", "be", "As", "as", "  as", "bring me", "bring  me"], # got, as, bring me
        "col2": ["got", "got", "be", "As", "  as", "bring me", "bring  me", "purple"], # got, as, bring me
        "col3": ["", "got", "as", "had", "pink", "red", "ppl", "rah"] # -
    }
    data2 = {
        "col1": ["got", "had", "had  ", "as", "As", "bring me", "bring me", "be", "Be", "BE"], # had, as, bring me, be
        "col2": ["had", "had ", "none", "blue", "BLUE", "blue", "yl", "yl", "yl", "yl"], # had, blue, yl
        "col3": ["had", "had ", "none", "blue", "BLUE", "red", "red", "red", "red", "red"] # had, blue, red
    }
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    sim_matrix, mask = top_k_sim(df1, df2, 50)
    expected = np.array([[2/5, 0, 0],
                [2/5, 0, 0],
                [0, 0, 0]])
    np.testing.assert_array_equal(sim_matrix, expected)


## Missing: different Data types, empty values and empty strings


