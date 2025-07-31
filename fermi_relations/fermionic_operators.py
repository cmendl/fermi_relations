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
    id2 = sparse.identity(2)
    z = sparse.csr_matrix([[ 1.,  0.], [ 0., -1.]])
    u = sparse.csr_matrix([[ 0.,  0.], [ 1.,  0.]])
    clist = []
    for i in range(nmodes):
        c = sparse.identity(1)
        for j in range(nmodes):
            if j < i:
                c = sparse.kron(c, id2)
            elif j == i:
                c = sparse.kron(c, u)
            else:
                c = sparse.kron(c, z)
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


def kinetic_exponential(nmodes: int, i: int, j: int, t: float):
    """
    Construct the unitary matrix exponential of the kinetic hopping term.
    """
    clist, alist, nlist = construct_fermionic_operators(nmodes)
    numop_proj = (nlist[i] @ (sparse.identity(2**nmodes) - nlist[j])
                + nlist[j] @ (sparse.identity(2**nmodes) - nlist[i]))
    tkin = clist[i] @ alist[j] + clist[j] @ alist[i]
    # Euler representation
    return sparse.identity(2**nmodes) + (np.cos(t) - 1) * numop_proj - 1j * np.sin(t) * tkin


def interaction_exponential(nmodes: int, i: int, j: int, t: float):
    """
    Construct the unitary matrix exponential of the interaction term (n_i - 1/2) (n_j - 1/2).
    """
    _, _, nlist = construct_fermionic_operators(nmodes)
    vint = (nlist[i] - 0.5*sparse.identity(2**nmodes)) @ (nlist[j] - 0.5*sparse.identity(2**nmodes))
    # Euler representation
    return np.cos(0.25*t) * sparse.identity(2**nmodes) - 4j * np.sin(0.25*t) * vint
