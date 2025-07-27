import unittest
import numpy as np
from scipy.linalg import expm, logm, schur, block_diag
from scipy import sparse
import scipy.sparse.linalg as spla
import fermi_relations as fr


class TestMajorana(unittest.TestCase):

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
                    self.assertEqual(spla.norm(anti_comm(mi, mj)
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


def anti_comm(a, b):
    """
    Anti-commutator {a, b} = a b + b a.
    """
    return a @ b + b @ a


def comm(a, b):
    """
    Commutator [a, b] = a b - b a.
    """
    return a @ b - b @ a


if __name__ == "__main__":
    unittest.main()
