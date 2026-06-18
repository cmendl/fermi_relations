"""
Test fermionic operators and related functions.
"""

import numpy as np
from scipy.linalg import expm
from scipy import sparse
import scipy.sparse.linalg as spla
from scipy.stats import unitary_group
import fermi_relations as fr


def test_total_number_op():
    """
    Test construction of the total number operator.
    """
    for nmodes in range(1, 8):
        _, _, nlist = fr.construct_fermionic_operators(nmodes)
        ntot = fr.total_number_op(nmodes)
        assert spla.norm(ntot - sum(nlist)) == 0


def test_givens_rotation():
    """
    Test matrix representation of a single-particle base change
    described by a Givens rotation.
    """
    rng = np.random.default_rng()

    # generalized Givens rotation
    gmat = unitary_group.rvs(2, random_state=rng)
    assert np.allclose(gmat.conj().T @ gmat, np.identity(2))

    # corresponding base change matrix on two-mode Fock space
    gfock = fr.fock_orbital_base_change(gmat).todense()

    # reference matrix
    gfock_ref = fr.orbital_rotation_gate(gmat)

    # compare
    assert np.allclose(gfock, gfock_ref)


def test_orbital_base_change():
    """
    Test matrix representation of a single-particle base change
    on overall Fock space.
    """
    rng = np.random.default_rng()

    # number of modes
    nmodes = 5

    # for a single-particle identity map, the overall base change matrix
    # should likewise be the identity map
    ufock = fr.fock_orbital_base_change(np.identity(nmodes))
    assert spla.norm(ufock - sparse.identity(2**nmodes)) == 0

    # random orthonormal states
    u = unitary_group.rvs(nmodes, random_state=rng)
    assert np.allclose(u.conj().T @ u, np.identity(nmodes))

    ufock = fr.fock_orbital_base_change(u)
    # must likewise be unitary
    assert spla.norm(ufock.conj().T @ ufock - sparse.identity(2**nmodes)) < 1e-13

    idx = [0, 2, 3]
    psi_ref = fr.SlaterDeterminant(u[:, idx]).to_vector()
    # encode indices in binary format
    i = sum(1 << (nmodes - j - 1) for j in idx)
    # need to reshape since slicing returns matrix (different from numpy convention)
    psi = np.reshape(ufock[:, i].toarray(), -1)
    # compare
    assert np.allclose(psi, psi_ref)

    # base change applied to creation and annihilation operators
    clist, alist, _ = fr.construct_fermionic_operators(nmodes)
    for i in range(nmodes):
        assert spla.norm(ufock @ clist[i] @ ufock.conj().T
                         - fr.orbital_create_op(u[:, i])) < 1e-13
        assert spla.norm(ufock @ alist[i] @ ufock.conj().T
                         - fr.orbital_annihil_op(u[:, i])) < 1e-13

    # compose two base changes
    v = unitary_group.rvs(nmodes, random_state=rng)
    vfock = fr.fock_orbital_base_change(v)
    assert spla.norm(ufock @ vfock
                     - fr.fock_orbital_base_change(u @ v)) < 1e-13


def test_skew_number_op():
    """
    Test action of a number operator w.r.t.
    a non-orthogonal basis state on a Slater determinant.
    """
    rng = np.random.default_rng()

    # number of modes
    nmodes = 7
    # number of particles
    nptcl = 3
    # random orthonormal states
    base = unitary_group.rvs(nmodes, random_state=rng)
    orb = base[:, :nptcl]

    # create state vector representation of a Slater determinant
    psi = fr.SlaterDeterminant(orb).to_vector()

    # non-orthogonal state
    x = fr.crandn(nmodes, rng)
    x /= np.linalg.norm(x)
    n = fr.orbital_number_op(x)

    # manually construct sum of Slater determinants by projecting one orbital at a time onto 'x'
    n_psi = 0
    for i in range(nptcl):
        n_orbs_i = np.concatenate((orb[:, :i],
                                   np.reshape(np.vdot(x, orb[:, i]) * x, (nmodes, 1)),
                                   orb[:, i+1:]), axis=1)
        n_psi += fr.SlaterDeterminant(n_orbs_i).to_vector()
    # compare
    assert np.allclose(n_psi, n @ psi)


def test_free_fermion_hamiltonian():
    """
    Test relations of a free-fermion Hamiltonian.
    """
    rng = np.random.default_rng()

    # number of modes
    nmodes = 7

    # random single-particle Hamiltonian
    h = fr.crandn((nmodes, nmodes), rng)
    h = 0.5*(h + h.conj().T)
    # Hamiltonian on the whole Fock space
    clist, alist, _ = fr.construct_fermionic_operators(nmodes)
    hfock = sum(h[i, j] * (clist[i] @ alist[j]) for i in range(nmodes) for j in range(nmodes))

    # number of particles
    nptcl = 3
    # random orthonormal states
    base = unitary_group.rvs(nmodes, random_state=rng)
    orb = base[:, :nptcl]
    # create Slater determinant
    psi = fr.SlaterDeterminant(orb).to_vector()

    # energy expectation value
    en = np.vdot(psi, hfock.toarray() @ psi)
    assert abs(en - np.trace(orb.conj().T @ h @ orb)) < 1e-14

    # time evolution operator on the whole Fock space
    ufock = expm(-1j*hfock.toarray())

    # time-evolved state
    psi_t = ufock @ psi
    # alternative construction: time-evolve single-particle states individually
    orb_t = expm(-1j*h) @ orb
    psi_t_alt = fr.SlaterDeterminant(orb_t).to_vector()
    # compare
    assert np.allclose(psi_t_alt, psi_t)

    # express the time evolution operator in terms of "orbital" number operators
    # diagonalize 'h'
    eigvals, eigvecs = np.linalg.eigh(h)
    nlist_orb = [fr.orbital_number_op(x) for x in eigvecs.T]
    ufock_alt = np.identity(2**nmodes)
    for i in range(nmodes):
        ufock_alt = ufock_alt @ ((np.identity(2**nmodes) - nlist_orb[i])
                                + np.exp(-1j*eigvals[i]) * nlist_orb[i])
    # compare
    assert np.allclose(ufock_alt, ufock)


def test_thouless_theorem():
    """
    Numerically verify Thouless' theorem.
    """
    rng = np.random.default_rng()

    # number of modes
    nmodes = 5

    # random single-particle base change matrix as matrix exponential
    h = fr.crandn((nmodes, nmodes), rng)
    # identity even holds if 'h' is not Hermitian (and 'u' not unitary)
    # h = 0.5*(h + h.conj().T)
    u = expm(-1j*h)

    clist, alist, _ = fr.construct_fermionic_operators(nmodes)
    tfock = sum(h[i, j] * (clist[i] @ alist[j]) for i in range(nmodes) for j in range(nmodes))
    ufock = expm(-1j*tfock.toarray())

    # reference base change matrix on the whole Fock space
    ufock_ref = fr.fock_orbital_base_change(u)

    # compare
    assert np.allclose(ufock, ufock_ref.toarray())


def test_kinetic_exponential():
    """
    Test properties of the matrix exponential of the kinetic hopping term.
    """
    t = 0.7
    for nmodes in range(2, 8):
        clist, alist, _ = fr.construct_fermionic_operators(nmodes)
        for i in range(nmodes):
            for j in range(nmodes):
                if i == j:
                    continue
                tkin = clist[i] @ alist[j] + clist[j] @ alist[i]
                ufock_ref = expm(-1j * t * tkin.toarray())
                ufock = fr.kinetic_exponential(nmodes, i, j, t)
                assert np.allclose(ufock.toarray(), ufock_ref)


def test_hubbard_interaction_exponential():
    """
    Test properties of the matrix exponential of the Hubbard model interaction term.
    """
    t = 0.4
    for nmodes in range(1, 8):
        _, _, nlist = fr.construct_fermionic_operators(nmodes)
        for i in range(nmodes):
            for j in range(nmodes):
                vint = ((nlist[i].toarray() - 0.5*np.identity(2**nmodes))
                      @ (nlist[j].toarray() - 0.5*np.identity(2**nmodes)))
                ufock_ref = expm(-1j * t * vint)
                ufock = fr.hubbard_interaction_exponential(nmodes, i, j, t)
                assert np.allclose(ufock.toarray(), ufock_ref)
