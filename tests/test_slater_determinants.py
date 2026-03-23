import numpy as np
from scipy.stats import unitary_group
import scipy.sparse.linalg as spla
from scipy import sparse
import fermi_relations as fr


def test_total_number_op():
    """
    Test construction of the total number operator.
    """
    for nmodes in range(1, 8):
        _, _, nlist = fr.construct_fermionic_operators(nmodes)
        ntot = fr.total_number_op(nmodes)
        assert spla.norm(ntot - sum(nlist)) == 0


def test_slater_determinant():
    """
    Test Slater determinant construction.
    """
    rng = np.random.default_rng()

    # number of modes
    nmodes = 7
    # number of particles
    nptcl = 3
    # random orthonormal states
    base = unitary_group.rvs(nmodes, random_state=rng)
    orb = base[:, :nptcl]
    # create Slater determinant
    psi = fr.slater_determinant(orb)
    # must be normalized
    assert abs(np.linalg.norm(psi) - 1) < 1e-13
    # must be eigenstate of number operator
    assert np.linalg.norm(fr.total_number_op(nmodes) @ psi - nptcl*psi) < 1e-13
    for i in range(nmodes):
        # number operator of an individual mode
        n = fr.orbital_number_op(base[:, i])
        assert np.linalg.norm(n @ psi - (1 if i < nptcl else 0) * psi) < 1e-13


def test_vdot_slater():
    """
    Test inner product of two Slater determinants.
    """
    rng = np.random.default_rng()

    # number of modes
    nmodes = 8
    # number of particles
    nptcl = 5
    # random orthonormal states
    chi = unitary_group.rvs(nmodes, random_state=rng)[:, :nptcl]
    phi = unitary_group.rvs(nmodes, random_state=rng)[:, :nptcl]
    # compare with inner product of state vectors (as reference)
    assert abs(fr.vdot_slater(chi, phi)
               - np.vdot(fr.slater_determinant(chi), fr.slater_determinant(phi))) < 1e-13


def test_orthonormalize_slater_determinant():
    """
    Test Slater determinant orthonormalization.
    """
    rng = np.random.default_rng()

    # number of modes
    nmodes = 7
    # number of particles
    nptcl = 4
    # non-orthonormalized orbitals
    orb = 0.5 * fr.crandn((nmodes, nptcl), rng)
    psi_ref = fr.slater_determinant(orb)
    # orthonormalize
    orb_orth, overlap = fr.orthonormalize_slater_determinant(orb)
    psi = overlap * fr.slater_determinant(orb_orth)
    # compare
    assert np.allclose(psi, psi_ref)


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
    psi_ref = fr.slater_determinant(u[:, idx])
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

    # create Slater determinant
    psi = fr.slater_determinant(orb)

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
        n_psi += fr.slater_determinant(n_orbs_i)
    # compare
    assert np.allclose(n_psi, n @ psi)
