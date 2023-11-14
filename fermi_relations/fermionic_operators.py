from functools import cache
import numpy as np
from scipy import sparse


@cache
def construct_fermionic_operators(nmodes: int):
    """
    Generate sparse matrix representations of the fermionic creation and
    annihilation operators for `nmodes` modes (or sites),
    based on Jordan-Wigner transformation.
    """
    I = sparse.identity(2)
    Z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    U = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    clist = []
    for i in range(nmodes):
        c = sparse.identity(1)
        for j in range(nmodes):
            if j < i:
                c = sparse.kron(c, I)
            elif j == i:
                c = sparse.kron(c, U)
            else:
                c = sparse.kron(c, Z)
        c = sparse.csr_matrix(c)
        c.eliminate_zeros()
        clist.append(c)
    # corresponding annihilation operators
    alist = [sparse.csr_matrix(c.conj().T) for c in clist]
    # corresponding number operators
    nlist = []
    for i in range(nmodes):
        f = 1 << (nmodes - i - 1)
        data = [1. if (n & f == f) else 0. for n in range(2**nmodes)]
        nlist.append(sparse.dia_matrix((data, 0), 2*(2**nmodes,)))
    return clist, alist, nlist


def orbital_create_op(x):
    """
    Fermionic "orbital" creation operator.
    """
    x = np.asarray(x)
    nmodes = len(x)
    clist, _, _ = construct_fermionic_operators(nmodes)
    return sum(x[i] * clist[i] for i in range(nmodes))


def orbital_annihil_op(x):
    """
    Fermionic "orbital" annihilation operator.
    """
    # anti-linear with respect to coefficients in `x`
    return orbital_create_op(x).conj().T


def orbital_number_op(x):
    """
    Fermionic "orbital" number operator.
    """
    c = orbital_create_op(x)
    return c @ c.conj().T


def total_number_op(nmodes: int):
    """
    Total number operator on full Fock space.
    """
    data = np.array([n.bit_count() for n in range(2**nmodes)], dtype=float)
    ind = np.arange(2**nmodes)
    return sparse.csr_matrix((data, (ind, ind)), shape=(2**nmodes, 2**nmodes))
