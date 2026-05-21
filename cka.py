import torch


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA similarity between activation matrices X (n, p) and Y (n, q).

    CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Both matrices are column-centred before computation.
    Returns a scalar in [0, 1]: 1 = identical geometry, 0 = orthogonal.
    """
    X = X.float().cpu()
    Y = Y.float().cpu()

    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    num = (XtY ** 2).sum()
    denom = torch.norm(XtX, 'fro') * torch.norm(YtY, 'fro')

    if denom < 1e-10:
        return float('nan')

    return float(num / denom)
