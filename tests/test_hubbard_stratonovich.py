import unittest
import numpy as np
from scipy.stats import unitary_group
import fermi_relations as fr


class TestHubbardStratonovich(unittest.TestCase):

    def test_discrete_hubbard_stratonovich(self):
        """
        Test the discrete Hubbard-Stratonovich transformation of the Hubbard interaction
        applied to a Slater determinant, see Eq. (6) in Phys. Rev. B 40, 506 (1989).

        Reference:
            S. R. White, D. J. Scalapino, R. L. Sugar, E. Y. Loh, J. E. Gubernatis, R. T. Scalettar
            Numerical study of the two-dimensional Hubbard model
            Phys. Rev. B 40, 506 (1989)
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 7

        # create a random Slater determinant
        # number of particles
        nptcl = 5
        # random orthonormal states
        base = unitary_group.rvs(nmodes, random_state=rng)
        orb = base[:, :nptcl]
        psi = fr.slater_determinant(orb)

        t = 0.8

        # note: mu is complex-valued in general
        mu = np.arccos(np.exp(0.5j * t))
        # absolute values of 'u' and 'v' are not 1 in general since 'mu' is complex
        u = np.exp( 1j * mu)
        v = np.exp(-1j * mu)

        for i in range(nmodes):
            for j in range(nmodes):
                if i == j:
                    continue

                # time-evolved state governed by the Hubbard interaction term, as reference
                psi_t_ref = fr.hubbard_interaction_exponential(nmodes, i, j, t) @ psi

                # use Hubbard-Stratonovich transformation
                # to represent time-evolved state as a sum of two Slater determinants
                a = np.identity(nmodes, dtype=complex)
                b = np.identity(nmodes, dtype=complex)
                a[i, i] = u
                a[j, j] = v
                b[i, i] = v
                b[j, j] = u
                psi_t = 0.5 * np.exp(-0.25j * t) * (
                      fr.slater_determinant(a @ orb)
                    + fr.slater_determinant(b @ orb))

                # compare
                self.assertTrue(np.allclose(psi_t, psi_t_ref))


if __name__ == '__main__':
    unittest.main()
