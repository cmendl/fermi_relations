import unittest
import numpy as np
from scipy.stats import unitary_group
import scipy.sparse.linalg as spla
from scipy import sparse
import fermi_relations as fr


class TestFock(unittest.TestCase):

    def test_total_number_op(self):
        """
        Test construction of the total number operator.
        """
        for nmodes in range(1, 8):
            _, _, nlist = fr.construct_fermionic_operators(nmodes)
            ntot = fr.total_number_op(nmodes)
            self.assertEqual(spla.norm(ntot - sum(nlist)), 0)

    def test_slater_determinant(self):
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
        self.assertAlmostEqual(np.linalg.norm(psi), 1, delta=1e-13)
        # must be eigenstate of number operator
        self.assertAlmostEqual(np.linalg.norm(
            fr.total_number_op(nmodes) @ psi - nptcl*psi), 0, delta=1e-13)
        for i in range(nmodes):
            # number operator of an individual mode
            n = fr.orbital_number_op(base[:, i])
            self.assertAlmostEqual(np.linalg.norm(
                n @ psi - (1 if i < nptcl else 0) * psi), 0, delta=1e-13)

    def test_orthonormalize_slater_determinant(self):

        rng = np.random.default_rng()

        # number of modes
        nmodes = 7
        # number of particles
        nptcl = 4
        # non-orthonormalized orbitals
        orb = 0.5 * rng.standard_normal((nmodes, nptcl))
        psi_ref = fr.slater_determinant(orb)
        # orthonormalize
        orb_orth, overlap = fr.orthonormalize_slater_determinant(orb)
        psi = overlap * fr.slater_determinant(orb_orth)
        # compare
        self.assertTrue(np.allclose(psi, psi_ref))

    def test_orbital_base_change(self):
        """
        Test matrix representation of single-particle base change
        on overall Fock space.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 5

        # for a single-particle identity map, the overall base change matrix
        # should likewise be the identity map
        ufull = fr.fock_orbital_base_change(np.identity(nmodes))
        self.assertEqual(spla.norm(ufull - sparse.identity(2**nmodes)), 0)

        # random orthonormal states
        u = unitary_group.rvs(nmodes, random_state=rng)
        self.assertTrue(np.allclose(u.conj().T @ u, np.identity(nmodes)))

        ufull = fr.fock_orbital_base_change(u)
        # must likewise be unitary
        self.assertAlmostEqual(spla.norm(
            ufull.conj().T @ ufull - sparse.identity(2**nmodes)), 0, delta=1e-13)

        idx = [0, 2, 3]
        psi_ref = fr.slater_determinant(u[:, idx])
        # encode indices in binary format
        i = sum(1 << (nmodes - j - 1) for j in idx)
        # need to reshape since slicing returns matrix (different from numpy convention)
        psi = np.reshape(ufull[:, i].toarray(), -1)
        # compare
        self.assertTrue(np.allclose(psi, psi_ref))

    def test_skew_number_op(self):
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
        self.assertTrue(np.allclose(n_psi, n @ psi))


if __name__ == '__main__':
    unittest.main()
