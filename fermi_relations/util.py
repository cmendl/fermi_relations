import numpy as np


def anti_comm(a, b):
    """
    Anti-commutator {a, b} = a b + b a.
    """
    return a @ b + b @ a


def comm(a, b):
    """
    Commutator [a, b] = a b - b a.
    """
    return a @ b - b @ a


def crandn(size=None, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


def gauss_hermite_quadrature(n: int):
    """
    Compute the Gauss-Hermite quadrature points and weights.
    """
    # Golub-Welsch algorithm
    entries = np.sqrt(np.arange(1, n))
    points, vecs = np.linalg.eigh(np.diag(entries, 1) + np.diag(entries, -1))
    # require that eigenvectors are normalized
    weights = np.abs(vecs[0, :])**2
    return points, weights
