import numpy as np


def tensor_product(a):
    """
    Calculate the matrix representation of the n-fold tensor products of an operator.
    """
    a = np.asarray(a)
    nmodes = a.shape[1]
    assert a.shape == (nmodes, nmodes)
    a_fock = np.zeros((2**nmodes, 2**nmodes), dtype=a.dtype)
    for m in range(2**nmodes):
        m1 = _one_bit_indices(m)
        for n in range(2**nmodes):
            if m.bit_count() != n.bit_count():
                continue
            n1 = _one_bit_indices(n)
            # temporary matrix for computing determinant
            d = np.zeros((len(m1), len(n1)), dtype=a.dtype)
            for i, k in enumerate(m1):
                for j, l in enumerate(n1):
                    d[i, j] = a[nmodes - k - 1, nmodes - l - 1]
            a_fock[m, n] = np.linalg.det(d)
    return a_fock


def _one_bit_indices(n: int):
    """
    Determine the positions of the 1-bits in 'n'.
    """
    idx = []
    for i in range(n.bit_length()):
        if n & (1 << i):
            idx.append(i)
    return idx
