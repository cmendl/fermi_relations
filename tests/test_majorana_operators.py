import unittest
import numpy as np
from scipy.linalg import expm, logm, schur, block_diag
from scipy import sparse
import scipy.sparse.linalg as spla
import fermi_relations as fr


class TestMajoranaOperators(unittest.TestCase):

    def test_hermitian(self):
        """
        Verify that the Majorana operators are Hermitian.
        """
        for nmodes in range(1, 8):
            mlist = fr.construct_majorana_operators(nmodes)
            self.assertEqual(len(mlist), 2*nmodes)
            for m in mlist:
                self.assertEqual(spla.norm(m - m.conj().T), 0)

    def test_anti_comm(self):
        """
        Verify that {mi, mj} == 2 * delta_{ij}.
        """
        for nmodes in range(1, 8):
            mlist = fr.construct_majorana_operators(nmodes)
            self.assertEqual(len(mlist), 2*nmodes)
            for i, mi in enumerate(mlist):
                for j, mj in enumerate(mlist):
                    delta = (1 if i == j else 0)
                    self.assertEqual(spla.norm(fr.anti_comm(mi, mj)
                                               - 2 * delta * sparse.identity(2**nmodes)), 0)

    def test_orthonormality(self):
        """
        Verify orthonormality with respect to the trace product.
        """
        for nmodes in range(1, 8):
            mlist = fr.construct_majorana_operators(nmodes)
            self.assertEqual(len(mlist), 2*nmodes)
            for i, mi in enumerate(mlist):
                for j, mj in enumerate(mlist):
                    delta = (1 if i == j else 0)
                    self.assertEqual((mi @ mj).trace() / 2**nmodes, delta)

    def test_kinetic_hopping(self):
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
                    self.assertEqual(spla.norm(tkin_maj - tkin_ref), 0)

    def test_number_op(self):
        """
        Express the fermionic number operator in terms of Majorana operators.
        """
        for nmodes in range(1, 8):
            _, _, nlist = fr.construct_fermionic_operators(nmodes)
            mlist = fr.construct_majorana_operators(nmodes)
            for i, n_ref in enumerate(nlist):
                n_maj = 0.5 * (sparse.identity(2**nmodes) + 1j * mlist[2*i] @ mlist[2*i+1])
                self.assertEqual(spla.norm(n_maj - n_ref), 0)

    def test_kinetic_exponential(self):
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
                    ufull_ref = expm(-1j * t * tkin.toarray())
                    ufull_maj = expm(0.5 * t * (mlist[2*i] @ mlist[2*j+1]
                                              + mlist[2*j] @ mlist[2*i+1]).toarray())
                    ufull_maj_alt = fr.kinetic_exponential_majorana(nmodes, i, j, t)
                    self.assertTrue(np.allclose(ufull_maj, ufull_ref))
                    self.assertTrue(np.allclose(ufull_maj_alt.toarray(), ufull_ref))

    def test_interaction_exponential(self):
        """
        Test properties of the matrix exponential of the interaction term
        based on Majorana operators.
        """
        t = 0.3
        for nmodes in range(1, 8):
            _, _, nlist = fr.construct_fermionic_operators(nmodes)
            for i in range(nmodes):
                for j in range(nmodes):
                    vint = ((nlist[i].toarray() - 0.5*np.identity(2**nmodes))
                          @ (nlist[j].toarray() - 0.5*np.identity(2**nmodes)))
                    ufull_ref = expm(-1j * t * vint)
                    ufull = fr.interaction_exponential_majorana(nmodes, i, j, t)
                    self.assertTrue(np.allclose(ufull.toarray(), ufull_ref))

    def test_free_fermion_hamiltonian(self):
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
            h = rng.standard_normal((2*nmodes, 2*nmodes))
            h = 0.5*(h - h.conj().T)
            # Hamiltonian on full Fock space, Eq. (11)
            mlist = fr.construct_majorana_operators(nmodes)
            hfull = 1j * sum(h[i, j] * (mlist[i] @ mlist[j])
                             for i in range(2*nmodes)
                             for j in range(2*nmodes))
            # must be Hermitian
            self.assertAlmostEqual(spla.norm(hfull.conj().T - hfull), 0., delta=1e-14)

            u = expm(4*h)
            ufull = expm(-1j*hfull.toarray())
            # must be unitary
            self.assertTrue(np.allclose(ufull.conj().T @ ufull, np.identity(ufull.shape[1])))

            # Majorana mode transformation, Eq. (12)
            for i, m in enumerate(mlist):
                self.assertTrue(
                    np.allclose(ufull.conj().T @ m @ ufull,
                                sum(u[i, j] * mlist[j].toarray() for j in range(len(mlist)))))

            # block-diagonalize 'h', Eq. (14)
            t, w = schur(h, output="real")
            if np.linalg.det(w) < 0:
                w[:, 0] = -w[:, 0]
                t[:, 0] = -t[:, 0]
                t[:, 1] = -t[:, 1]
            # 'w' must be in SO(2*n, R)
            self.assertTrue(np.isrealobj(w))
            self.assertTrue(np.allclose(w.T @ w, np.identity(w.shape[1])))
            self.assertTrue(np.allclose(np.linalg.det(w), 1))
            # must diagonalize 'h'
            self.assertTrue(np.allclose(h @ w, w @ t))
            # verify block-diagonal structure
            blocks = [t[i:i+2, i:i+2] for i in range(0, 2*nmodes, 2)]
            self.assertTrue(np.allclose(t, block_diag(*blocks)))
            lambda_list = np.array([b[1, 0] for b in blocks])
            self.assertEqual(len(lambda_list), nmodes)
            for l, b in zip(lambda_list, blocks):
                self.assertTrue(np.allclose(b, [[0, -l], [l, 0]]))

            # Eq. (15)
            g = logm(w) / 4
            self.assertTrue(np.isrealobj(g))
            # must be anti-symmetric
            self.assertTrue(np.allclose(g.T, -g))
            self.assertTrue(np.allclose(w, expm(4*g)))

            gfull = 1j * sum(g[i, j] * (mlist[i] @ mlist[j])
                             for i in range(2*nmodes)
                             for j in range(2*nmodes))
            # must be Hermitian
            self.assertAlmostEqual(spla.norm(gfull.conj().T - gfull), 0., delta=1e-13)
            wfull = expm(-1j*gfull.toarray())
            # must be unitary
            self.assertTrue(np.allclose(wfull.conj().T @ wfull, np.identity(wfull.shape[1])))

            # Eq. (17)
            hdiag = -2j * sum(lambda_list[i] * (mlist[2*i] @ mlist[2*i+1]) for i in range(nmodes))
            self.assertTrue(np.allclose(wfull.conj().T @ hfull @ wfull,
                                        hdiag.toarray()))


if __name__ == "__main__":
    unittest.main()
