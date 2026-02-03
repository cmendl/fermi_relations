import unittest
import numpy as np
from scipy.stats import unitary_group
import fermi_relations as fr


class TestReducedDensityMatrices(unittest.TestCase):
    """
    Test relations of reduced density matrices (RDMs).
    """

    def test_rdm1_slater(self):
        """
        Test properties of a one-body reduced density matrix
        originating from a Slater determinant.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 7

        # create a random Slater determinant
        # number of particles
        nptcl = 4
        # random orthonormal states
        base = unitary_group.rvs(nmodes, random_state=rng)
        orb = base[:, :nptcl]
        # create Slater determinant
        psi = fr.slater_determinant(orb)

        rho = fr.rdm1(nmodes, psi)

        # 'rho' must be Hermitian
        self.assertTrue(np.allclose(rho, rho.conj().T))

        # eigenvalues of the reduced density matrix of a Slater determinant must be 0 or 1
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals_ref = np.array((nmodes - nptcl) * [0] + nptcl * [1])
        self.assertTrue(np.allclose(eigvals, eigvals_ref))

        # "natural orbitals" consisting of eigenvectors corresponding to eigenvalue 1
        natural_orb = eigvecs[:, (nmodes - nptcl):]
        # natural orbitals must be a unitary linear combination of the original orbitals
        u = natural_orb.conj().T @ orb
        self.assertTrue(np.allclose(u.conj().T @ u, np.identity(u.shape[1])))

        psi_reconstr = fr.slater_determinant(natural_orb)
        # overlap should be 1 (up to a phase factor)
        self.assertAlmostEqual(abs(np.vdot(psi_reconstr, psi)), 1)
        self.assertAlmostEqual(abs(fr.vdot_slater(natural_orb, orb)), 1)

        # 'psi' must be an eigenstate with eigenvalue 1 of the natural orbitals number operators
        for norb in natural_orb.T:
            nop = fr.orbital_number_op(norb)
            self.assertTrue(np.allclose(nop @ psi, psi))

    def test_rdm1_generic(self):
        """
        Test properties of a one-body reduced density matrix
        originating from a generic quantum state.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 7

        # random normalized state
        psi = fr.crandn(2**nmodes, rng)
        psi /= np.linalg.norm(psi)

        rho = fr.rdm1(nmodes, psi)

        # 'rho' must be Hermitian
        self.assertTrue(np.allclose(rho, rho.conj().T))

        # eigenvalues of 'rho' must be in the interval [0, 1]
        eigvals = np.linalg.eigvalsh(rho)
        self.assertTrue(all(0 <= eigvals) and all(eigvals <= 1))

    def test_rdm2(self):
        """
        Test properties of a two-body reduced density matrix.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 7

        # random normalized state
        psi = fr.crandn(2**nmodes, rng)
        psi /= np.linalg.norm(psi)

        gamma = fr.rdm2(nmodes, psi)

        # 'gamma' must be Hermitian
        self.assertTrue(np.allclose(gamma, gamma.conj().T))

        # eigenvalues must be non-negative, i.e., gamma must be positive semidefinite
        eigvals = np.linalg.eigvalsh(gamma)
        self.assertTrue(all(0 <= eigvals))

    def test_expectation_value(self):
        """
        Verify that the expectation value of a one- and two-body Hamiltonian
        can be computed based on the respective reduced density matrices.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 7

        # random normalized state
        psi = fr.crandn(2**nmodes, rng)
        psi /= np.linalg.norm(psi)

        # reduced density matrices
        rho   = fr.rdm1(nmodes, psi)
        gamma = fr.rdm2(nmodes, psi)

        # Hamiltonian, does not need to be Hermitian for this test
        tkin = fr.crandn(2 * (nmodes,), rng)
        vint = fr.crandn(4 * (nmodes,), rng)
        h = construct_hamiltonian(tkin, vint)

        # evaluate expectation value based on reduced density matrices
        idxtuples = [(i, j) for i in range(nmodes) for j in range(i + 1, nmodes)]
        vint_eff = np.array([[
              vint[i[0], i[1], j[0], j[1]]
            - vint[i[1], i[0], j[0], j[1]]
            - vint[i[0], i[1], j[1], j[0]]
            + vint[i[1], i[0], j[1], j[0]]
            for j in idxtuples]
            for i in idxtuples])
        avr = np.trace(tkin @ rho) + 0.5 * np.trace(vint_eff @ gamma)

        # reference expectation value
        avr_ref = np.vdot(psi, h @ psi)

        # compare
        self.assertAlmostEqual(avr, avr_ref)

    def test_representability_conditions(self):
        """
        Test representability conditions.

        Reference:
            Zhengji Zhao, Bastiaan J. Braams, Mituhiro Fukuda,
            Michael L. Overton, Jerome K. Percus:
            The reduced density matrix method for electronic structure calculations
            and the role of three-index representability conditions
        	J. Chem. Phys. 120, 2095 (2004)
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 7

        # random normalized state
        psi = fr.crandn(2**nmodes, rng)
        psi /= np.linalg.norm(psi)

        # reduced density matrices
        rho   = fr.rdm1(nmodes, psi)
        gamma = fr.rdm2(nmodes, psi)

        clist, alist, _ = fr.construct_fermionic_operators(nmodes)

        # all tuples (i, j) with i < j
        idxtuples = [(i, j) for i in range(nmodes) for j in range(i + 1, nmodes)]

        # 'q' tensor, defined in terms of 'rho' and 'gamma'
        q = gamma + np.array([[
                - delta(i[0], j[0]) * rho[i[1], j[1]]
                - delta(i[1], j[1]) * rho[i[0], j[0]]
                + delta(i[0], j[1]) * rho[i[1], j[0]]
                + delta(i[1], j[0]) * rho[i[0], j[1]]
                + delta(i[0], j[0]) * delta(i[1], j[1])
                - delta(i[0], j[1]) * delta(i[1], j[0])
                               for j in idxtuples]
                               for i in idxtuples])
        # reference 'q' tensor
        q_ref = np.array([[np.vdot(psi,
                    alist[i[1]] @ (alist[i[0]] @ (clist[j[0]] @ (clist[j[1]] @ psi))))
                           for j in idxtuples]
                           for i in idxtuples])
        self.assertTrue(np.allclose(q, q_ref))
        # 'q' must be Hermitian
        self.assertTrue(np.allclose(q, q.conj().T))
        # eigenvalues must be non-negative, i.e., 'q' must be positive semidefinite
        self.assertTrue(all(0 <= np.linalg.eigvalsh(q)))

        gamma_full = np.zeros(4 * (nmodes,), dtype=gamma.dtype)
        for i, idx_i in enumerate(idxtuples):
            for j, idx_j in enumerate(idxtuples):
                val = gamma[i, j]
                gamma_full[idx_i[0], idx_i[1], idx_j[0], idx_j[1]] =  val
                gamma_full[idx_i[0], idx_i[1], idx_j[1], idx_j[0]] = -val
                gamma_full[idx_i[1], idx_i[0], idx_j[0], idx_j[1]] = -val
                gamma_full[idx_i[1], idx_i[0], idx_j[1], idx_j[0]] =  val

        # 'g' tensor, defined in terms of 'rho' and 'gamma'
        g = np.array([[gamma_full[i[0], j[1], i[1], j[0]] + delta(i[0], j[0]) * rho[j[1], i[1]]
                       for j in idxtuples]
                       for i in idxtuples])
        # reference 'g' tensor
        g_ref = np.array([[np.vdot(psi,
                    clist[i[1]] @ (alist[i[0]] @ (clist[j[0]] @ (alist[j[1]] @ psi))))
                           for j in idxtuples]
                           for i in idxtuples])
        self.assertTrue(np.allclose(g, g_ref))
        # 'g' must be Hermitian
        self.assertTrue(np.allclose(g, g.conj().T))
        # eigenvalues must be non-negative, i.e., 'g' must be positive semidefinite
        self.assertTrue(all(0 <= np.linalg.eigvalsh(g)))

        # all triples (i, j, k) with i < j < k
        idxtriples = [(i, j, k)
                      for i in range(nmodes)
                      for j in range(i + 1, nmodes)
                      for k in range(j + 1, nmodes)]

        # cyclic and anti-cyclic three-index permutations
        eps3idx = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (0, 2, 1), (1, 0, 2)]
        eps3sgn = [1, 1, 1, -1, -1, -1]

        # 't1' tensor, defined in terms of 'rho' and 'gamma'
        t1 = np.array([[sum(si * sj *  delta(i[pi[0]], j[pj[0]])
                                    * (delta(i[pi[1]], j[pj[1]])
                                       * (1/6. * delta(i[pi[2]], j[pj[2]])
                                          - 0.5 * rho[i[pi[2]], j[pj[2]]])
                                       + 0.25 * gamma_full[i[pi[1]], i[pi[2]], j[pj[1]], j[pj[2]]])
                            for sj, pj in zip(eps3sgn, eps3idx)
                            for si, pi in zip(eps3sgn, eps3idx))
                         for j in idxtriples]
                         for i in idxtriples])
        # reference 't1' tensor
        t1_ref = np.array([[np.vdot(psi,
                                    clist[j[2]] @
                                   (clist[j[1]] @
                                   (clist[j[0]] @
                                   (alist[i[0]] @
                                   (alist[i[1]] @
                                   (alist[i[2]] @ psi))))))
                          + np.vdot(psi,
                                    alist[i[0]] @
                                   (alist[i[1]] @
                                   (alist[i[2]] @
                                   (clist[j[2]] @
                                   (clist[j[1]] @
                                   (clist[j[0]] @ psi))))))
                           for j in idxtriples]
                           for i in idxtriples])
        self.assertTrue(np.allclose(t1, t1_ref))
        # 't1' must be Hermitian
        self.assertTrue(np.allclose(t1, t1.conj().T))
        # eigenvalues must be non-negative, i.e., 't1' must be positive semidefinite
        self.assertTrue(all(0 <= np.linalg.eigvalsh(t1)))

        # cyclic and anti-cyclic two-index permutations
        eps2idx = [(0, 1), (1, 0)]
        eps2sgn = [1, -1]

        # 't2' tensor, defined in terms of 'rho' and 'gamma'
        t2 = np.array([[sum(si * sj * (
              0.5  * rho[i[0], j[0]] * delta(i[1+pi[0]], j[1+pj[0]]) * delta(i[1+pi[1]], j[1+pj[1]])
            + 0.25 * delta(i[0], j[0]) * gamma_full[j[1+pj[0]], j[1+pj[1]], i[1+pi[0]], i[1+pi[1]]]
                   - delta(i[1+pi[0]], j[1+pj[0]]) * gamma_full[i[0], j[1+pj[1]], j[0], i[1+pi[1]]])
                            for sj, pj in zip(eps2sgn, eps2idx)
                            for si, pi in zip(eps2sgn, eps2idx))
                         for j in idxtriples]
                         for i in idxtriples])
        # reference 't2' tensor
        t2_ref = np.array([[np.vdot(psi,
                                    clist[i[2]] @
                                   (clist[i[1]] @
                                   (alist[i[0]] @
                                   (clist[j[0]] @
                                   (alist[j[1]] @
                                   (alist[j[2]] @ psi))))))
                          + np.vdot(psi,
                                    clist[j[0]] @
                                   (alist[j[1]] @
                                   (alist[j[2]] @
                                   (clist[i[2]] @
                                   (clist[i[1]] @
                                   (alist[i[0]] @ psi))))))
                           for j in idxtriples]
                           for i in idxtriples])
        self.assertTrue(np.allclose(t2, t2_ref))
        # 't2' must be Hermitian
        self.assertTrue(np.allclose(t2, t2.conj().T))
        # eigenvalues must be non-negative, i.e., 't2' must be positive semidefinite
        self.assertTrue(all(0 <= np.linalg.eigvalsh(t2)))


def delta(i, j):
    """
    Kronecker delta function.
    """
    return 1 if i == j else 0


def construct_hamiltonian(tkin, vint):
    """
    Construct an interacting fermionic Hamiltonian as a sparse matrix.
    """
    nmodes = tkin.shape[0]
    clist, alist, _ = fr.construct_fermionic_operators(nmodes)
    h = (
        # single-particle terms
        sum(tkin[i, j] * (clist[i] @ alist[j])
             for j in range(nmodes)
             for i in range(nmodes)) +
         # interaction terms; note ordering of 'k' and 'l'
         sum(0.5 * vint[i, j, k, l] * (clist[i] @ clist[j] @ alist[l] @ alist[k])
             for l in range(nmodes)
             for k in range(nmodes)
             for j in range(nmodes)
             for i in range(nmodes)))
    h.eliminate_zeros()
    return h


if __name__ == '__main__':
    unittest.main()
