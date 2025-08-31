import unittest
import numpy as np
import scipy.sparse.linalg as spla
import fermi_relations as fr


class TestDoubleFactorization(unittest.TestCase):

    def test_double_factorization(self):
        """
        Test the double-factorized form of a molecular Hamiltonian interaction term.

        Reference:
            Mario Motta, Erika Ye, Jarrod R. McClean, Zhendong Li,
            Austin J. Minnich, Ryan Babbush, Garnet Kin-Lic Chan
            Low rank representations for quantum simulation of electronic structure
        	npj Quantum Inf. 7, 83 (2021)
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 6

        vint = _random_molecular_interaction_coefficients(nmodes, 0.1, rng)

        # check symmetries
        self.assertTrue(np.allclose(vint.transpose(1, 0, 2, 3), vint))
        self.assertTrue(np.allclose(vint.transpose(0, 1, 3, 2), vint))
        self.assertTrue(np.allclose(vint.transpose(2, 3, 0, 1), vint))
        vmat = vint.reshape((nmodes**2, nmodes**2))
        self.assertTrue(np.allclose(vmat.T, vmat))

        lmat = _truncated_sqrtm(vmat)
        self.assertTrue(np.isrealobj(lmat))
        self.assertTrue(np.allclose(lmat @ lmat.T, vmat))
        ltensor = lmat.reshape((nmodes, nmodes, -1))
        # symmetry must be preserved
        self.assertTrue(np.allclose(ltensor.transpose((1, 0, 2)), ltensor))

        vop_ref = fr.molecular_interaction(vint, physics_convention=False)

        # Eq. (6)
        clist, alist, nlist = fr.construct_fermionic_operators(nmodes)
        lops = []
        for l in range(ltensor.shape[-1]):
            h = ltensor[:, :, l]
            self.assertTrue(np.allclose(h.T, h))
            lops.append(sum(h[i, j] * (clist[i] @ alist[j])
                            for i in range(nmodes)
                            for j in range(nmodes)))
        vop_squares = 0.5 * sum(lop @ lop for lop in lops)
        # compare
        self.assertAlmostEqual(spla.norm(vop_squares - vop_ref), 0, delta=1e-14)

        # Eq. (3)
        vop_number = 0
        for l in range(ltensor.shape[-1]):
            eig, u = np.linalg.eigh(ltensor[:, :, l])
            nlist = []
            for i in range(len(eig)):
                # orbital number operator w.r.t. new basis
                nlist.append(fr.orbital_number_op(u[:, i]))
            for i in range(len(eig)):
                for j in range(len(eig)):
                    # only number operators from the same 'nlist' (common basis) commute
                    self.assertAlmostEqual(spla.norm(fr.comm(nlist[i], nlist[j])), 0, delta=1e-14)
                    vop_number += 0.5 * eig[i] * eig[j] * (nlist[i] @ nlist[j])
        # compare
        self.assertAlmostEqual(spla.norm(vop_number - vop_ref), 0, delta=1e-14)


def _truncated_sqrtm(a, tol: float = 1e-14):
    """
    Construct the truncated matrix square root `x`
    of the positive-semidefinite input matrix `a`
    such that `a ≈ x @ x.T`.
    """
    eigvals, u = np.linalg.eigh(a)
    # reverse to sort eigenvalues by magnitude in descending order
    idx = np.where(eigvals > tol)[0][::-1]
    return u[:, idx] * np.sqrt(eigvals[idx])


def _random_molecular_interaction_coefficients(nmodes: int, tau: float, rng: np.random.Generator):
    """
    Construct a random molecular interaction coefficient tensor using chemists' convention.
    """
    x = rng.standard_normal((nmodes, nmodes, nmodes**2))
    x = x * np.exp(-tau * np.arange(0, nmodes**2))
    x /= np.linalg.norm(x)
    # note: symmetrization implies that `x` does not have full rank
    x = 0.5 * (x + x.transpose((1, 0, 2)))
    # ensure that `vint` is positive semidefinite (after grouping dimensions (0, 1) amd (2, 3))
    vint = np.einsum(x, (0, 1, 4), x, (2, 3, 4), (0, 1, 2, 3))
    return vint


if __name__ == '__main__':
    unittest.main()
