import numpy as np
from scipy.linalg import expm, logm, schur, block_diag
from scipy import sparse
import scipy.sparse.linalg as spla
import fermi_relations as fr


def test_majorana_hermitian():
    """
    Verify that the Majorana operators are Hermitian.
    """
    for nmodes in range(1, 8):
        mlist = fr.construct_majorana_operators(nmodes)
        assert len(mlist) == 2*nmodes
        for m in mlist:
            assert spla.norm(m - m.conj().T) == 0


def test_majorana_anti_comm():
    """
    Verify that {mi, mj} == 2 * delta_{ij}.
    """
    for nmodes in range(1, 8):
        mlist = fr.construct_majorana_operators(nmodes)
        assert len(mlist) == 2*nmodes
        for i, mi in enumerate(mlist):
            for j, mj in enumerate(mlist):
                delta = (1 if i == j else 0)
                assert spla.norm(fr.anti_comm(mi, mj)
                                 - 2 * delta * sparse.identity(2**nmodes)) == 0


def test_majorana_orthonormality():
    """
    Verify orthonormality with respect to the trace product.
    """
    for nmodes in range(1, 8):
        mlist = fr.construct_majorana_operators(nmodes)
        assert len(mlist) == 2*nmodes
        for i, mi in enumerate(mlist):
            for j, mj in enumerate(mlist):
                delta = (1 if i == j else 0)
                assert (mi @ mj).trace() / 2**nmodes == delta


def test_majorana_kinetic_hopping():
    """
    Express the kinetic hopping operator in terms of Majorana operators.
    """
    for nmodes in range(2, 8):
        clist, alist, _ = fr.construct_fermionic_operators(nmodes)
        mlist = fr.construct_majorana_operators(nmodes)
        for i in range(nmodes):
            for j in range(nmodes):
                if i == j:
                    continue
                tkin_ref = clist[i] @ alist[j] + clist[j] @ alist[i]
                tkin_maj = 0.5j * (mlist[2*i] @ mlist[2*j+1] + mlist[2*j] @ mlist[2*i+1])
                assert spla.norm(tkin_maj - tkin_ref) == 0


def test_majorana_number_op():
    """
    Express the fermionic number operator in terms of Majorana operators,
    `n_j = 1/2 (I + i m_{2j} m_{2j+1})`.
    """
    for nmodes in range(1, 8):
        _, _, nlist = fr.construct_fermionic_operators(nmodes)
        mlist = fr.construct_majorana_operators(nmodes)
        for i, n_ref in enumerate(nlist):
            n_maj = 0.5 * (sparse.identity(2**nmodes) + 1j * mlist[2*i] @ mlist[2*i+1])
            assert spla.norm(n_maj - n_ref) == 0


def test_majorana_kinetic_exponential():
    """
    Test the matrix exponential representation of the kinetic hopping term
    based on Majorana operators.
    """
    t = 1.3
    for nmodes in range(2, 8):
        clist, alist, _ = fr.construct_fermionic_operators(nmodes)
        mlist = fr.construct_majorana_operators(nmodes)
        for i in range(nmodes):
            for j in range(nmodes):
                if i == j:
                    continue
                tkin = clist[i] @ alist[j] + clist[j] @ alist[i]
                ufock_ref = expm(-1j * t * tkin.toarray())
                ufock_maj = expm(0.5 * t * (mlist[2*i] @ mlist[2*j+1]
                                          + mlist[2*j] @ mlist[2*i+1]).toarray())
                ufock_maj_alt = fr.kinetic_exponential_majorana(nmodes, i, j, t)
                assert np.allclose(ufock_maj, ufock_ref)
                assert np.allclose(ufock_maj_alt.toarray(), ufock_ref)


def test_majorana_hubbard_interaction_exponential():
    """
    Test properties of the matrix exponential of the Hubbard model interaction term
    based on Majorana operators.
    """
    t = 0.3
    for nmodes in range(1, 8):
        _, _, nlist = fr.construct_fermionic_operators(nmodes)
        for i in range(nmodes):
            for j in range(nmodes):
                vint = ((nlist[i].toarray() - 0.5*np.identity(2**nmodes))
                      @ (nlist[j].toarray() - 0.5*np.identity(2**nmodes)))
                ufock_ref = expm(-1j * t * vint)
                ufock = fr.hubbard_interaction_exponential_majorana(nmodes, i, j, t)
                assert np.allclose(ufock.toarray(), ufock_ref)


def test_majorana_hubbard_interaction_exponential_conjugation():
    """
    Test the equation describing the conjugation by the matrix exponential
    of the Hubbard model interaction term based on Majorana operators.
    """
    t = 0.7
    for nmodes in range(1, 8):
        mlist = fr.construct_majorana_operators(nmodes)
        for i in range(nmodes):
            for j in range(nmodes):
                # ensure that i != j
                if i == j:
                    continue
                ufock = fr.hubbard_interaction_exponential_majorana(nmodes, i, j, t)
                for k, m in enumerate(mlist):
                    mconj_ref = ufock.conj().T @ m @ ufock
                    idx = [2*i, 2*i+1, 2*j, 2*j+1]
                    if k not in idx:
                        mconj = m
                    else:
                        idx.remove(k)
                        mconj = (np.cos(0.5*t) * m
                                 + 1j * np.sin(0.5*t) * (-1)**k
                                 * (mlist[idx[0]] @ mlist[idx[1]] @ mlist[idx[2]]))
                    assert spla.norm(mconj - mconj_ref) < 1e-14


def test_majorana_free_fermion_hamiltonian():
    """
    Test relations of a free-fermion Hamiltonian in the Majorana representation.

    Reference:
        Adrian Chapman and Steven T. Flammia
        Characterization of solvable spin models via graph invariants
        Quantum 4, 278 (2020)
        arXiv:2003.05465
    """
    rng = np.random.default_rng()

    # number of modes
    for nmodes in range(1, 8):
        # random real anti-symmetric single-particle Hamiltonian
        h = fr.antisymmetrize(rng.standard_normal((2*nmodes, 2*nmodes)))
        # Hamiltonian on the whole Fock space, Eq. (11)
        mlist = fr.construct_majorana_operators(nmodes)
        hfock = 1j * sum(h[i, j] * (mlist[i] @ mlist[j])
                         for i in range(2*nmodes)
                         for j in range(2*nmodes))
        # must be Hermitian
        assert spla.norm(hfock.conj().T - hfock) < 1e-14

        u = expm(4*h)
        ufock = expm(-1j*hfock.toarray())
        # must be unitary
        assert np.allclose(ufock.conj().T @ ufock, np.identity(ufock.shape[1]))

        # Majorana mode transformation, Eq. (12)
        for i, m in enumerate(mlist):
            assert np.allclose(ufock.conj().T @ m @ ufock,
                               sum(u[i, j] * mlist[j].toarray() for j in range(len(mlist))))

        # block-diagonalize 'h', Eq. (14)
        t, w = schur(h, output="real")
        if np.linalg.det(w) < 0:
            w[:, 0] = -w[:, 0]
            t[:, 0] = -t[:, 0]
            t[:, 1] = -t[:, 1]
        # 'w' must be in SO(2*n, R)
        assert np.isrealobj(w)
        assert np.allclose(w.T @ w, np.identity(w.shape[1]))
        assert np.allclose(np.linalg.det(w), 1)
        # must diagonalize 'h'
        assert np.allclose(h @ w, w @ t)
        # verify block-diagonal structure
        blocks = [t[i:i+2, i:i+2] for i in range(0, 2*nmodes, 2)]
        assert np.allclose(t, block_diag(*blocks))
        lambda_list = np.array([b[1, 0] for b in blocks])
        assert len(lambda_list) == nmodes
        for l, b in zip(lambda_list, blocks):
            assert np.allclose(b, [[0, -l], [l, 0]])

        # Eq. (15)
        g = logm(w) / 4
        assert np.isrealobj(g)
        # must be anti-symmetric
        assert np.allclose(g.T, -g)
        assert np.allclose(w, expm(4*g))

        gfock = 1j * sum(g[i, j] * (mlist[i] @ mlist[j])
                         for i in range(2*nmodes)
                         for j in range(2*nmodes))
        # must be Hermitian
        assert spla.norm(gfock.conj().T - gfock) < 1e-13
        wfock = expm(-1j*gfock.toarray())
        # must be unitary
        assert np.allclose(wfock.conj().T @ wfock, np.identity(wfock.shape[1]))

        # Eq. (17)
        hdiag = -2j * sum(lambda_list[i] * (mlist[2*i] @ mlist[2*i+1]) for i in range(nmodes))
        assert np.allclose(wfock.conj().T @ hfock @ wfock,
                           hdiag.toarray())

        # use diagonalization for matrix exponential
        ufock_alt = np.identity(ufock.shape[0])
        # transpose 'w' for inverse transformation
        mlist_h = [sum(w[j, i] * mlist[j].toarray()
                       for j in range(len(mlist))) for i in range(2*nmodes)]
        for i in range(nmodes):
            ufock_i = (np.cos(2*lambda_list[i]) * np.identity(ufock.shape[0])
                     - np.sin(2*lambda_list[i]) * (mlist_h[2*i] @ mlist_h[2*i+1]))
            ufock_alt = ufock_alt @ ufock_i
        assert np.allclose(ufock, ufock_alt)


def test_majorana_base_change():
    """
    Test relations of single-mode base changes.
    """
    rng = np.random.default_rng()

    # number of bases
    nbases = 3

    # number of modes
    for nmodes in range(1, 8):

        mlist = fr.construct_majorana_operators(nmodes)

        # random real anti-symmetric single-particle Hamiltonians
        hlist = [fr.antisymmetrize(rng.standard_normal((2*nmodes, 2*nmodes)))
                 for _ in range(nbases)]
        # Hamiltonians on the whole Fock space
        hfock_list = [1j * sum(h[i, j] * (mlist[i] @ mlist[j])
                               for i in range(2*nmodes)
                               for j in range(2*nmodes)) for h in hlist]
        # must be Hermitian
        for hfock in hfock_list:
            assert spla.norm(hfock.conj().T - hfock) < 1e-14

        # single-mode base change
        u = np.identity(2*nmodes)
        for h in hlist:
            u = u @ expm(4*h)
        # must be orthogonal
        assert np.isrealobj(u)
        assert np.allclose(u.T @ u, np.identity(2*nmodes))
        # base change on the whole Fock space
        ufock = np.identity(2**nmodes)
        for hfock in hfock_list:
            ufock = ufock @ expm(-1j*hfock.toarray())
        # must be unitary
        assert np.allclose(ufock.conj().T @ ufock, np.identity(ufock.shape[1]))

        # Majorana mode transformation
        mlist_new = [sum(u[i, j] * mlist[j].toarray() for j in range(len(mlist)))
                     for i in range(len(mlist))]
        for i, m in enumerate(mlist):
            assert np.allclose(ufock.conj().T @ m @ ufock, mlist_new[i])

        # basis transformation applied to a string of Majorana operators
        idx = rng.choice(2*nmodes, size=5)
        mstring = np.identity(2**nmodes)
        for i in idx:
            mstring = mstring @ mlist[i]
        mstring_new = np.identity(2**nmodes)
        for i in idx:
            mstring_new = mstring_new @ mlist_new[i]
        assert np.allclose(ufock.conj().T @ mstring @ ufock, mstring_new)

        # basis transformation applied to a matrix exponential of Majorana operators
        vint = 0.1 * fr.crandn(3 * (2*nmodes,), rng)
        op = expm(sum(vint[i, j, k] * (mlist[i] @ mlist[j] @ mlist[k]).toarray()
                      for i in range(len(mlist))
                      for j in range(len(mlist))
                      for k in range(len(mlist))))
        op_new = expm(sum(vint[i, j, k] * (mlist_new[i] @ mlist_new[j] @ mlist_new[k])
                          for i in range(len(mlist_new))
                          for j in range(len(mlist_new))
                          for k in range(len(mlist_new))))
        assert np.allclose(ufock.conj().T @ op @ ufock, op_new)
        # alternative calculation
        vint_new = np.einsum(u, (3, 0), u, (4, 1), u, (5, 2), vint, (3, 4, 5), (0, 1, 2))
        op_new_alt = expm(sum(vint_new[i, j, k] * (mlist[i] @ mlist[j] @ mlist[k]).toarray()
                              for i in range(len(mlist))
                              for j in range(len(mlist))
                              for k in range(len(mlist))))
        assert np.allclose(ufock.conj().T @ op @ ufock, op_new_alt)


def test_majorana_string_basis():
    """
    Test properties of the Majorana string basis.
    """
    # number of modes
    for nmodes in range(1, 5):
        mbasis = fr.construct_majorana_string_basis(nmodes)
        # must be unitary up to a scaling factor
        assert spla.norm(mbasis.conj().T @ mbasis
                         - 2**nmodes * sparse.identity(4**nmodes)) == 0
        # first string must be the identity matrix
        assert spla.norm(mbasis[:, 0].reshape(2 * (2**nmodes,))
                         - sparse.identity(2**nmodes)) == 0
        # check Hermitian property of each Majorana string
        for v in mbasis.T:
            mstring = v.reshape(2 * (2**nmodes,))
            assert spla.norm(mstring.conj().T - mstring) == 0


def test_majorana_strings_commute():
    """
    Test decision function whether two Majorana strings commute.
    """
    # number of modes
    for nmodes in range(1, 5):
        mbasis = fr.construct_majorana_string_basis(nmodes)
        for ima in range(2**(2*nmodes)):
            # corresponding Majorana string
            msa = mbasis[:, ima].reshape(2 * (2**nmodes,))
            for imb in range(2**(2*nmodes)):
                # corresponding Majorana string
                msb = mbasis[:, imb].reshape(2 * (2**nmodes,))
                comm_ref = spla.norm(fr.comm(msa, msb)) == 0
                assert fr.majorana_strings_commute(ima, imb) == comm_ref


def test_majorana_string_products():
    """
    Test evaluation of products of Majorana strings.
    """
    # number of modes
    for nmodes in range(1, 5):
        mbasis = fr.construct_majorana_string_basis(nmodes)
        for ima in range(2**(2*nmodes)):
            # corresponding Majorana string
            msa = mbasis[:, ima].reshape(2 * (2**nmodes,))
            for imb in range(2**(2*nmodes)):
                # corresponding Majorana string
                msb = mbasis[:, imb].reshape(2 * (2**nmodes,))
                # evaluate product based on encoded bit representations
                imp, phase = fr.majorana_string_product(ima, imb)
                msp = phase * mbasis[:, imp].reshape(2 * (2**nmodes,))
                # compare
                assert spla.norm(msp - msa @ msb) == 0
