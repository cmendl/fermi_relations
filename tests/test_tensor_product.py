import unittest
import numpy as np
from scipy.stats import unitary_group
import fermi_relations as fr


class TestTensorProduct(unittest.TestCase):

    def test_unitary_tensor_product(self):
        """
        Test matrix representation of the n-fold tensor products of an operator
        by comparing with a unitary base change.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 7

        # random orthonormal matrix
        u = unitary_group.rvs(nmodes, random_state=rng)
        u_fock_ref = fr.fock_orbital_base_change(u)

        u_tensor = fr.tensor_product(u)

        # compare
        self.assertTrue(np.allclose(u_tensor, u_fock_ref.todense()))

    def test_twofold_tensor_product(self):
        """
        Test matrix representation of the two-fold tensor products of an operator.
        """
        rng = np.random.default_rng()

        # number of modes
        nmodes = 6

        # random single-site operator
        a = fr.crandn((nmodes, nmodes), rng)
        # tensor products on the whole Fock space
        a_tensor = fr.tensor_product(a)

        # random orthonormal states
        phi = unitary_group.rvs(nmodes, random_state=rng)[:, :2]

        psi2 = a_tensor @ fr.slater_determinant(phi)

        # reference calculation
        psi = a @ phi
        psi2_ref = []
        for i in reversed(range(nmodes)):
            for j in reversed(range(i + 1, nmodes)):
                psi2_ref.append((psi[i, 0] * psi[j, 1] - psi[i, 1] * psi[j, 0]))
        psi2_ref = np.asarray(psi2_ref)

        # compare
        idx = [i for i in range(2**nmodes) if i.bit_count() == 2]
        self.assertTrue(np.allclose(psi2[idx], psi2_ref))


if __name__ == '__main__':
    unittest.main()
