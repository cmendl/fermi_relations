from fermi_relations.fermiops import construct_fermionic_operators
import numpy as np


def rdm1(nmodes: int, psi):
    """
    Compute the one-body reduced density matrix
    of a given quantum state 'psi' on the whole Fock space.
    """
    psi = np.asarray(psi)
    assert psi.shape == (2**nmodes,)
    clist, alist, _ = construct_fermionic_operators(nmodes)
    return np.array([[np.vdot(psi, cj @ (ai @ psi))
                      for cj in clist]
                      for ai in alist])


def rdm2(nmodes: int, psi):
    """
    Compute the two-body reduced density matrix
    of a given quantum state 'psi' on the whole Fock space.
    """
    psi = np.asarray(psi)
    assert psi.shape == (2**nmodes,)
    clist, alist, _ = construct_fermionic_operators(nmodes)
    # all tuples (i, j) with i < j
    idxtuples = [(i, j) for i in range(nmodes) for j in range(i + 1, nmodes)]
    return np.array([[np.vdot(psi, clist[j[0]] @ (clist[j[1]] @ (alist[i[1]] @ (alist[i[0]] @ psi))))
                      for j in idxtuples]
                      for i in idxtuples])
