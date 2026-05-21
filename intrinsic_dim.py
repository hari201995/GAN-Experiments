import torch


def two_nn_id(X: torch.Tensor) -> float:
    """
    Two-NN intrinsic dimensionality estimator (Facco et al. 2017).

    For each point, compute mu_i = dist(2nd-NN) / dist(1st-NN).
    Under the two-NN Pareto model the MLE gives:
        ID = (n - 1) / sum(log(mu_i))

    X: (n, d) activation matrix — one row per sample.
    Returns a scalar ID estimate, or nan when n < 3 or geometry is degenerate.
    """
    n = X.shape[0]
    if n < 3:
        return float('nan')

    X = X.float().cpu()
    dists = torch.cdist(X, X)           # (n, n)
    dists.fill_diagonal_(float('inf'))
    sorted_d, _ = dists.sort(dim=1)
    r1 = sorted_d[:, 0]                 # distance to 1st-NN
    r2 = sorted_d[:, 1]                 # distance to 2nd-NN

    valid = (r1 > 0) & (r2 > r1)
    if valid.sum() < 3:
        return float('nan')

    mu = r2[valid] / r1[valid]
    n_v = int(valid.sum())
    return float((n_v - 1) / torch.log(mu).sum().item())
