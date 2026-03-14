import numpy as np
import pandas as pd
import pytest
from app.similarity.schema.type import dtype_to_category, find_type_similarity, type_compat

def test_dtype_to_category_integer():
    assert dtype_to_category(np.int32) == "Integer"
    assert dtype_to_category(np.int64) == "Integer"

def test_dtype_to_category_decimal():
    assert dtype_to_category(np.float32) == "Decimal"
    assert dtype_to_category(np.float64) == "Decimal"

def test_dtype_to_category_boolean():
    assert dtype_to_category(np.bool_) == "Boolean"

def test_dtype_to_category_string():
    assert dtype_to_category(np.dtype("U10")) == "String"

def test_dtype_to_category_object():
    assert dtype_to_category(object) == "Object"

def test_dtype_to_category_datetime():
    assert dtype_to_category(np.dtype("datetime64[ns]")) == "Date"

def test_dtype_to_category_timedelta():
    assert dtype_to_category(np.dtype("timedelta64[ns]")) == "Timedelta"

def test_type_compat_symmetry():
    assert type_compat[("Integer","Decimal")] == type_compat[("Decimal","Integer")]

def test_find_type_similarity_basic():
    df1 = pd.DataFrame({"a":[1,2,3],"b":[1.0,2.0,3.0]})
    df2 = pd.DataFrame({"c":[4,5,6],"d":[True,False,True]})
    sim, mask = find_type_similarity(df1, df2)
    assert sim.shape == (2,2)
    assert mask.shape == (2,2)
    assert sim[0,0] == 1
    assert sim[1,0] == 0.8

def test_find_type_similarity_identical_types():
    df1 = pd.DataFrame({"a":[1,2,3]})
    df2 = pd.DataFrame({"b":[4,5,6]})
    sim, mask = find_type_similarity(df1, df2)
    assert sim[0,0] == 1
    assert mask[0,0] == 1

def test_find_type_similarity_string_object():
    df1 = pd.DataFrame({"a":["x","y"]})
    df2 = pd.DataFrame({"b":[object(),object()]})
    sim, mask = find_type_similarity(df1, df2)
    assert sim[0,0] == type_compat[("String","Object")]

def test_find_type_similarity_date_types():
    df1 = pd.DataFrame({"a":pd.date_range("2020-01-01", periods=3)})
    df2 = pd.DataFrame({"b":pd.date_range("2021-01-01", periods=3)})
    sim, mask = find_type_similarity(df1, df2)
    assert sim[0,0] == 1

def test_find_type_similarity_empty_dataframes():
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    sim, mask = find_type_similarity(df1, df2)
    assert sim.shape == (0,0)
    assert mask.shape == (0,0)

def test_find_type_similarity_mixed_types_matrix():
    df1 = pd.DataFrame({"a":[1],"b":[1.5],"c":[True]})
    df2 = pd.DataFrame({"d":["x"],"e":[pd.Timestamp("2020-01-01")]})
    sim, mask = find_type_similarity(df1, df2)
    assert sim.shape == (3,2)
    assert mask.shape == (3,2)
    assert sim[0,0] == type_compat[("Integer","String")]
    assert sim[1,1] == type_compat[("Decimal","Date")]