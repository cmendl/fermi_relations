import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from fermi_relations.fermionic_operators import orbital_create_op


def slater_determinant(phi):
    """
    Create the Slater determinant of single-particle states stored as column vectors in `phi`.
    """
    phi = np.asarray(phi)
    nmodes = phi.shape[0]
    # vacuum state
    psi = np.zeros(2**nmodes)
    psi[0] = 1
    for i in range(phi.shape[1]):
        psi = orbital_create_op(phi[:, i]) @ psi
    return psi


def orthonormalize_slater_determinant(phi):
    """
    Orthonormalize the states defining a Slater determinant.
    """
    chi, _ = np.linalg.qr(phi, mode="reduced")
    overlap = np.linalg.det(phi.conj().T @ chi)
    return chi, overlap


def fock_orbital_base_change(u):
    """
    Construct the matrix representation of a unitary, single-particle
    base change matrix described by `u` on the full Fock space.
    """
    u = np.asarray(u)
    nmodes = u.shape[1]
    clist = [orbital_create_op(u[:, i]) for i in range(nmodes)]
    u_fock = lil_matrix((2**nmodes, 2**nmodes), dtype=u.dtype)
    for m in range(2**nmodes):
        # vacuum state
        psi = np.zeros(2**nmodes)
        psi[0] = 1
        for i in range(nmodes):
            if m & (1 << (nmodes - i - 1)):
                psi = clist[i] @ psi
        u_fock[m] = psi
    return csr_matrix(u_fock.T)
