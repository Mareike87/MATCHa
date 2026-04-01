import numpy as np
import pytest

from app.matching.matcher import get_matches


def test_basic_match():
    attr1 = ["a", "b"]
    attr2 = ["x", "y"]
    sim_matrix = [
        [0.9, 0.1],
        [0.2, 0.8],
    ]
    result = get_matches(attr1, attr2, sim_matrix, threshold=0.8)
    assert result == [
        ("a", "x", 0.9),
        ("b", "y", 0.8),
    ]

def test_sorted_by_similarity_descending():
    attr1 = ["a", "b"]
    attr2 = ["x", "y"]
    sim_matrix = [
        [0.85, 0.95],
        [0.1, 0.2],
    ]
    result = get_matches(attr1, attr2, sim_matrix, threshold=0.8)
    assert result == [
        ("a", "y", 0.95),
        ("a", "x", 0.85),
    ]

def test_threshold_filters_values():
    attr1 = ["a", "b"]
    attr2 = ["x", "y"]
    sim_matrix = [
        [0.7, 0.6],
        [0.4, 0.9],
    ]
    result = get_matches(attr1, attr2, sim_matrix, threshold=0.8)
    assert result == [
        ("b", "y", 0.9),
    ]

def test_no_matches_returns_empty_list():
    attr1 = ["a", "b"]
    attr2 = ["x", "y"]
    sim_matrix = [
        [0.1, 0.2],
        [0.3, 0.4],
    ]
    result = get_matches(attr1, attr2, sim_matrix, threshold=0.8)
    assert result == []

def test_multiple_matches_same_row():
    attr1 = ["a"]
    attr2 = ["x", "y", "z"]
    sim_matrix = [
        [0.9, 0.85, 0.2],
    ]
    result = get_matches(attr1, attr2, sim_matrix, threshold=0.8)
    assert result == [
        ("a", "x", 0.9),
        ("a", "y", 0.85),
    ]

def test_empty_similarity_matrix():
    attr1 = []
    attr2 = []
    sim_matrix = []
    result = get_matches(attr1, attr2, sim_matrix, threshold=0.5)
    assert result == []

def test_threshold_equal_value_is_included():
    attr1 = ["a"]
    attr2 = ["x"]
    sim_matrix = [[0.8]]
    result = get_matches(attr1, attr2, sim_matrix, threshold=0.8)
    assert result == [("a", "x", 0.8)]

def test_all_values_above_threshold():
    attr1 = ["a", "b"]
    attr2 = ["x", "y"]
    sim_matrix = [
        [0.9, 0.95],
        [0.85, 0.88],
    ]
    result = get_matches(attr1, attr2, sim_matrix, threshold=0.8)
    expected = [
        ("a", "y", 0.95),
        ("a", "x", 0.9),
        ("b", "y", 0.88),
        ("b", "x", 0.85),
    ]
    assert result == expected

def test_same_similarity_values():
    attr1 = ["a", "b"]
    attr2 = ["x"]
    sim_matrix = [
        [0.9],
        [0.9],
    ]
    result = get_matches(attr1, attr2, sim_matrix, threshold=0.8)
    assert len(result) == 2
    assert all(match[2] == 0.9 for match in result)

def test_numpy_array_input():
    attr1 = ["a", "b"]
    attr2 = ["x", "y"]
    sim_matrix = np.array([
        [0.1, 0.9],
        [0.8, 0.2],
    ])
    result = get_matches(attr1, attr2, sim_matrix, threshold=0.8)
    assert result == [
        ("a", "y", 0.9),
        ("b", "x", 0.8),
    ]

def test_inconsistent_dimensions_raises_error():
    attr1 = ["a"]
    attr2 = ["x"]
    sim_matrix = [
        [0.9, 0.8],  # mehr Spalten als attr2
    ]
    with pytest.raises(IndexError):
        get_matches(attr1, attr2, sim_matrix, threshold=0.5)