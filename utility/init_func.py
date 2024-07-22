import numpy as np

def orthogonal_(matrix, random_state=None):
    """
    Fill the input `tensor` with a (semi) orthogonal matrix.

    Args:
        tensor: an n-dimensional `numpy.ndarray`, where `n >= 2`
        random_state: numpy random state for reproducibility (default: None)

    Returns:
        tensor: the updated `tensor` filled with (semi) orthogonal matrix
    """
    if matrix.ndim < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    # Get random state
    rng = np.random.default_rng(random_state)

    rows, cols = matrix.shape
    flattened = rng.standard_normal((rows, cols))

    if rows < cols:
        flattened = flattened.T

    # Compute the qr factorization
    q, r = np.linalg.qr(flattened)
    # Make Q uniform
    d = np.diag(r, 0)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T

    # matrix.view(q.shape).copy_(q)

    return q
