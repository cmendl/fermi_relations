"""
Reduced density matrices.
"""

from copy import deepcopy
import numpy as np
from fermi_relations.fermiops import construct_fermionic_operators
from fermi_relations.slater import SlaterDeterminant, vdot_slater


def quadratic_fermionic_average(nmodes: int, chi, psi):
    """
    Evaluate the matrix `<chi | a_j^{dagger} a_i | phi>_{ij}`
    for two quantum states 'chi' and 'psi' on the whole Fock space.
    """
    chi = np.asarray(chi)
    psi = np.asarray(psi)
    assert chi.shape == (2**nmodes,)
    assert psi.shape == (2**nmodes,)
    clist, alist, _ = construct_fermionic_operators(nmodes)
    return np.array([[np.vdot(chi, cj @ (ai @ psi))
                      for cj in clist]
                      for ai in alist])


def rdm1(nmodes: int, psi):
    """
    Compute the one-body reduced density matrix
    of a given quantum state 'psi' on the whole Fock space.
    """
    return quadratic_fermionic_average(nmodes, psi, psi)


def quadratic_fermionic_average_slater(chi: SlaterDeterminant, psi: SlaterDeterminant):
    """
    Evaluate the matrix `<chi | a_j^{dagger} a_i | phi>_{ij}`
    for two Slater determinants 'chi' and 'psi'.
    """
    return _quadratic_fermionic_average_slater_sum([chi], psi)


def _quadratic_fermionic_average_slater_sum(chi_list: list[SlaterDeterminant], psi: SlaterDeterminant):
    """
    Evaluate the matrix `sum_k <chi_k | a_j^{dagger} a_i | phi>_{ij}`
    for a list of Slater determinants 'chi_k' and a Slater determinant 'psi'.
    """
    assert all(chi.nmodes == psi.nmodes for chi in chi_list)
    nmodes = psi.nmodes
    # retain Slater determinants with matching particle number
    chi_list = [chi for chi in chi_list if chi.nptcl == psi.nptcl]
    nptcl = psi.nptcl
    if psi.is_orthonormal():
        psi_orth = psi
    else:
        psi_orth = deepcopy(psi)
        psi_orth.orthonormalize_orbitals()
    # temporarily switch to the orbital basis of 'psi'
    basis = np.linalg.qr(psi_orth.phi, mode="complete")[0]
    basis[:, :nptcl] = psi_orth.phi
    assert basis.shape == (nmodes, nmodes)
    # must be unitary
    assert np.allclose(basis.conj().T @ basis, np.identity(basis.shape[1]))
    rho = np.array(
        [[0 if i >= nptcl else
          sum(vdot_slater(chi, SlaterDeterminant(
            np.concatenate((basis[:, :i],
                            np.reshape(basis[:, j], (nmodes, 1)),
                            basis[:, i+1:nptcl]), axis=1), psi_orth.coeff)) for chi in chi_list)
          for j in range(nmodes)]
          for i in range(nmodes)])
    return basis @ rho @ basis.conj().T


def rdm1_slater_sum(psi: list[SlaterDeterminant]):
    """
    Compute the one-body reduced density matrix
    of the quantum state 'psi' represented as a sum of Slater determinants.
    """
    return sum(_quadratic_fermionic_average_slater_sum(psi, x) for x in psi)


def project_natural_orbitals(psi_list: list[SlaterDeterminant]) -> SlaterDeterminant:
    """
    Project a sum of Slater determinants onto the subspace spanned by
    the most relevant (largest eigenvalues) natural orbitals.
    """
    assert len(psi_list) > 0
    nptcl = psi_list[0].nptcl
    rho = rdm1_slater_sum(psi_list)
    _, eigvecs = np.linalg.eigh(rho)
    # eigenvalues are sorted in ascending order, so we take the last 'nptcl' eigenvectors
    norb_state = SlaterDeterminant(eigvecs[:, -nptcl:])
    coeff = sum(vdot_slater(norb_state, psi) for psi in psi_list)
    return SlaterDeterminant(norb_state.phi, coeff)


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
    return np.array(
        [[np.vdot(psi, clist[j[0]] @ (clist[j[1]] @ (alist[i[1]] @ (alist[i[0]] @ psi))))
            for j in idxtuples]
            for i in idxtuples])
