import unittest
import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group
import fermi_relations as fr


class TestFermionicOperators(unittest.TestCase):

    def test_free_fermion_hamiltonian(self):
        """
        Test relations of a free-fermion Hamiltonian.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 7

        # random single-particle Hamiltonian
        h = fr.crandn((nmodes, nmodes), rng)
        h = 0.5*(h + h.conj().T)
        # Hamiltonian on full Fock space
        clist, alist, _ = fr.construct_fermionic_operators(nmodes)
        hfull = sum(h[i, j] * (clist[i] @ alist[j]) for i in range(nmodes) for j in range(nmodes))

        # number of particles
        nptcl = 3
        # random orthonormal states
        base = unitary_group.rvs(nmodes, random_state=rng)
        orb = base[:, :nptcl]
        # create Slater determinant
        psi = fr.slater_determinant(orb)

        # energy expectation value
        en = np.vdot(psi, hfull.toarray() @ psi)
        self.assertAlmostEqual(en, np.trace(orb.conj().T @ h @ orb))

        # time-evolved state
        psi_t = expm(-1j*hfull.toarray()) @ psi
        # alternative construction: time-evolve single-particle states individually
        orb_t = expm(-1j*h) @ orb
        psi_t_alt = fr.slater_determinant(orb_t)
        # compare
        self.assertTrue(np.allclose(psi_t_alt, psi_t))

    def test_thouless_theorem(self):
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
        tfull = sum(h[i, j] * (clist[i] @ alist[j]) for i in range(nmodes) for j in range(nmodes))
        ufull = expm(-1j*tfull.toarray())

        # reference base change matrix on full Fock space
        ufull_ref = fr.fock_orbital_base_change(u)

        # compare
        self.assertTrue(np.allclose(ufull, ufull_ref.toarray()))

    def test_kinetic_exponential(self):
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
                    ufull_ref = expm(-1j * t * tkin.toarray())
                    ufull = fr.kinetic_exponential(nmodes, i, j, t)
                    self.assertTrue(np.allclose(ufull.toarray(), ufull_ref))

    def test_hubbard_interaction_exponential(self):
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
                    ufull_ref = expm(-1j * t * vint)
                    ufull = fr.hubbard_interaction_exponential(nmodes, i, j, t)
                    self.assertTrue(np.allclose(ufull.toarray(), ufull_ref))


if __name__ == '__main__':
    unittest.main()
