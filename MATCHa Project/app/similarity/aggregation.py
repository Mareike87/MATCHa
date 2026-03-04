import numpy as np

# aggregates set of n x m x l-matrices into n x m via max value
def aggregate_sim_max(sim_matrices):
    # Achtung! Hier darauf achten, entlang der richtigen Dimension zu reduzieren! (später kontrollieren)
    return np.max(sim_matrices, axis=2)

# aggregates set of n x m x l-matrices into n x m via mean of top k values
def aggregate_sim_top_k(sim_matrices, k):
    # Achtung! Hier darauf achten, entlang der richtigen Dimension zu reduzieren! (später kontrollieren)
    top_k = np.partition(sim_matrices, -k, axis=2)[:,:,-k:]
    return np.mean(top_k, axis=2)

# combines a number of sim matrices via weighted sum
def combine_sims_weighted(sim_matrices, masks, weights=None, clip=True):
    sim_matrices = np.asarray(sim_matrices)
    # If only one matrix is given transform shape to allow computation
    if sim_matrices.ndim == 2:
        sim_matrices = sim_matrices[None, :, :]
    # If no weights are given set equal weights
    if weights is None:
        weights = np.full(sim_matrices.shape[0], 1/sim_matrices.shape[0])
    else:
        weights = np.asarray(weights)
        # If weights do not sum to 1 set equal weights
        if not np.isclose(sum(weights), 1):
            # TODO: Add exception handling here
            print("Since weights do not sum up to 1 equal weights will be used by default.")
            weights = np.full(sim_matrices.shape[0], 1/sim_matrices.shape[0])

    weighted = sim_matrices * masks * weights[:, None, None]
    result_sim = weighted.sum(axis=0)
    # Clip to [0,1] since float calculations may lead to values outside this range
    if clip:
        result_sim = np.clip(result_sim, 0, 1)
    return result_sim

# TODO: hier noch masks einbauen
def combine_sims_var(sim_matrices, masks, clip=True):
    variances = np.array([np.var(sim) for sim in sim_matrices])
    masks = np.asarray(masks)
    sim_matrices = np.asarray(sim_matrices)
    weights = variances / np.sum(variances)
    print(weights)
    weighted = sim_matrices * masks * weights[:, None, None]
    result_sim = weighted.sum(axis=0)
    if clip:
        result_sim = np.clip(result_sim, 0, 1)
    return result_sim

def combine_sims_var_topk(sim_matrices, k, clip=True):
    vars = np.array([np.var(sim) for sim in sim_matrices])
