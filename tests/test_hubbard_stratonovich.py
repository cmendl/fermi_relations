import unittest
import numpy as np
from scipy.linalg import expm
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

    def test_continuous_hubbard_stratonovich(self):
        """
        Test the continuous Hubbard-Stratonovich transformation based on a Gaussian distribution
        and approximate the integral by Gauss-Hermite quadrature.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 6

        clist, alist, _ = fr.construct_fermionic_operators(nmodes)

        # random single-particle operator (not Hermitian in general)
        b = 0.5 * fr.crandn((nmodes, nmodes), rng)
        # free-fermion operator on full Fock space
        bfull = sum(b[i, j] * (clist[i] @ alist[j])
                    for i in range(nmodes)
                    for j in range(nmodes)).toarray()

        # squared operator
        vfull = bfull @ bfull

        t = 0.3

        # reference time evolution operator
        ufull_ref = expm(-0.5j * t * vfull)

        # continuous Hubbard-Stratonovich transformation
        phase = np.exp(-1j * np.pi / 4)
        points, weights = gauss_hermite_quadrature(15)
        ufull = sum(w * expm(-phase * x * np.sqrt(t) * bfull) for x, w in zip(points, weights))

        # compare
        self.assertTrue(np.allclose(ufull, ufull_ref, rtol=1e-13))


def gauss_hermite_quadrature(n: int):
    """
    Compute the Gauss-Hermite quadrature points and weights.
    """
    # Golub-Welsch algorithm
    entries = np.sqrt(np.arange(1, n))
    points, vecs = np.linalg.eigh(np.diag(entries, 1) + np.diag(entries, -1))
    # require that eigenvectors are normalized
    weights = np.abs(vecs[0, :])**2
    return points, weights


if __name__ == '__main__':
    unittest.main()
