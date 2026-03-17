import numpy as np
import pytest
from app.similarity.aggregation import aggregate_sim_max, aggregate_sim_top_k, combine_sims_weighted, combine_sims_var

def test_aggregate_sim_max_basic():
    sims = np.array([[[0.1,0.5,0.3],[0.2,0.4,0.6]],[[0.7,0.2,0.1],[0.3,0.8,0.2]]])
    sims = np.transpose(sims,(1,2,0))
    result = aggregate_sim_max(sims)
    expected = np.max(sims,axis=2)
    np.testing.assert_array_equal(result,expected)

def test_aggregate_sim_max_single_layer():
    sims = np.random.rand(3,4,1)
    result = aggregate_sim_max(sims)
    np.testing.assert_array_equal(result,sims[:,:,0])

def test_aggregate_sim_top_k_basic():
    sims = np.array([[[0.1,0.5,0.3],[0.2,0.4,0.6]],[[0.7,0.2,0.1],[0.3,0.8,0.2]]])
    sims = np.transpose(sims,(1,2,0))
    result = aggregate_sim_top_k(sims,1)
    expected = np.max(sims,axis=2)
    np.testing.assert_array_equal(result,expected)

def test_aggregate_sim_top_k_mean_of_two():
    sims = np.array([[[0.1,0.5,0.3],[0.2,0.4,0.6]]])
    sims = np.transpose(sims,(1,2,0))
    sims = np.concatenate([sims,sims],axis=2)
    result = aggregate_sim_top_k(sims,2)
    expected = np.mean(np.partition(sims,-2,axis=2)[:,:,-2:],axis=2)
    np.testing.assert_array_equal(result,expected)

def test_combine_sims_weighted_equal_weights():
    sims = np.array([[[0.5,0.5],[0.5,0.5]],[[1.0,1.0],[1.0,1.0]]])
    masks = np.ones_like(sims)
    result = combine_sims_weighted(sims,masks)
    expected = np.array([[0.75,0.75],[0.75,0.75]])
    np.testing.assert_allclose(result,expected)

def test_combine_sims_weighted_custom_weights():
    sims = np.array([[[0.0,0.0],[0.0,0.0]],[[1.0,1.0],[1.0,1.0]]])
    masks = np.ones_like(sims)
    result = combine_sims_weighted(sims,masks,weights=[0.2,0.8])
    expected = np.full((2,2),0.8)
    np.testing.assert_allclose(result,expected)

def test_combine_sims_weighted_invalid_weights_fallback():
    sims = np.array([[[0.0,0.0]],[[1.0,1.0]]])
    masks = np.ones_like(sims)
    result = combine_sims_weighted(sims,masks,weights=[1,1])
    expected = np.array([[0.5,0.5]])
    np.testing.assert_allclose(result,expected)

def test_combine_sims_weighted_single_matrix():
    sim = np.array([[0.2,0.3],[0.4,0.5]])
    mask = np.ones((1,2,2))
    result = combine_sims_weighted(sim,mask)
    np.testing.assert_array_equal(result,sim)

def test_combine_sims_weighted_mask_blocks_values():
    sims = np.array([[[1,1],[1,1]],[[0,0],[0,0]]])
    masks = np.array([[[1,1],[1,1]],[[0,0],[0,0]]])
    result = combine_sims_weighted(sims,masks)
    np.testing.assert_array_equal(result,np.array([[0.5,0.5],[0.5,0.5]]))

def test_combine_sims_var_basic_shape():
    sims = np.random.rand(3,2,2)
    masks = np.ones_like(sims)
    result = combine_sims_var(sims,masks)
    assert result.shape == (2,2)

def test_combine_sims_var_zero_mask():
    sims = np.array([[[0.2,0.3],[0.4,0.5]]])
    masks = np.zeros_like(sims)
    result = combine_sims_var(sims,masks)
    np.testing.assert_array_equal(result,np.zeros((2,2)))

def test_combine_sims_var_clipping():
    sims = np.array([[[2.0,2.0],[2.0,2.0]]])
    masks = np.ones_like(sims)
    result = combine_sims_var(sims,masks)
    assert np.all(result <= 1)