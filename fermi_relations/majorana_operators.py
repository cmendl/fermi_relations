from fermi_relations import construct_fermionic_operators
import numpy as np
from scipy import sparse


def construct_majorana_operators(nmodes: int):
    """
    Generate sparse matrix representations of the fermionic Majorana operators
    for `nmodes` modes (or sites).
    """
    clist, alist, _ = construct_fermionic_operators(nmodes)
    mlist = [[c + a, 1j*(c - a)] for c, a in zip(clist, alist)]
    return [m for mtuple in mlist for m in mtuple]


def kinetic_exponential_majorana(nmodes: int, i: int, j: int, t: float):
    """
    Construct the unitary matrix exponential of the kinetic hopping term
    based on Majorana operators.
    """
    mlist = construct_majorana_operators(nmodes)
    numop_proj = 0.5 * (sparse.identity(2**nmodes)
                        + mlist[2*i] @ mlist[2*i+1] @ mlist[2*j] @ mlist[2*j+1])
    itkin = -0.5 * (mlist[2*i] @ mlist[2*j+1] + mlist[2*j] @ mlist[2*i+1])
    # Euler representation
    return sparse.identity(2**nmodes) + (np.cos(t) - 1) * numop_proj - np.sin(t) * itkin


def interaction_exponential_majorana(nmodes: int, i: int, j: int, t: float):
    """
    Construct the unitary matrix exponential of the interaction term (n_i - 1/2) (n_j - 1/2)
    based on Majorana operators.
    """
    mlist = construct_majorana_operators(nmodes)
    vint = -mlist[2*i] @ mlist[2*i+1] @ mlist[2*j] @ mlist[2*j+1]
    # Euler representation
    return np.cos(0.25*t) * sparse.identity(2**nmodes) - 1j * np.sin(0.25*t) * vint
