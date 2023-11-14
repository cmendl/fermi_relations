import unittest
import numpy as np
from scipy import sparse
import scipy.sparse.linalg as spla
import fermi_relations as fr


class TestCommutators(unittest.TestCase):

    def test_anti_comm_ad_a(self):
        """
        Verify that {ai†, aj} == delta_{ij}.
        """
        for nmodes in range(1, 8):
            clist, alist, _ = fr.construct_fermionic_operators(nmodes)
            for i in range(nmodes):
                for j in range(nmodes):
                    ci = clist[i]
                    cj = clist[j]
                    ai = alist[i]
                    aj = alist[j]
                    delta = (1 if i == j else 0)
                    self.assertEqual(spla.norm(anti_comm(ci, aj)
                                               - delta * sparse.identity(2**nmodes)), 0)
                    self.assertEqual(spla.norm(anti_comm(ci, cj)), 0)
                    self.assertEqual(spla.norm(anti_comm(ai, aj)), 0)
        # use a numerical "orbital"
        rng = np.random.default_rng()
        nmodes = 5
        x = fr.crandn(nmodes, rng)
        x /= np.linalg.norm(x)
        c = fr.orbital_create_op(x)
        a = fr.orbital_annihil_op(x)
        self.assertAlmostEqual(spla.norm(anti_comm(c, a) - sparse.identity(2**nmodes)), 0, delta=1e-14)

    def test_comm_n_a(self):
        """
        Verify that [n, a†] == a† and [a, n] == a.
        """
        for nmodes in range(1, 8):
            clist, alist, nlist = fr.construct_fermionic_operators(nmodes)
            for i in range(nmodes):
                c = clist[i]
                a = alist[i]
                n = nlist[i]
                self.assertEqual(spla.norm(comm(n, c) - c), 0)
                self.assertEqual(spla.norm(comm(a, n) - a), 0)
        # use a numerical "orbital"
        rng = np.random.default_rng()
        x = fr.crandn(5, rng)
        x /= np.linalg.norm(x)
        a = fr.orbital_annihil_op(x)
        n = fr.orbital_number_op(x)
        self.assertAlmostEqual(spla.norm(comm(n, a.conj().T) - a.conj().T), 0)
        self.assertAlmostEqual(spla.norm(comm(a, n) - a), 0)

    def test_comm_n_hop(self):
        """
        Verify that [ni, ai† aj + aj† ai] == ai† aj - aj† ai.
        """
        for nmodes in range(1, 8):
            _, alist, nlist = fr.construct_fermionic_operators(nmodes)
            for i in range(nmodes):
                for j in range(nmodes):
                    if i == j:
                        continue
                    ni = nlist[i]
                    ai = alist[i]
                    aj = alist[j]
                    self.assertEqual(spla.norm(
                        comm(ni, ai.T @ aj + aj.T @ ai)
                          - (ai.T @ aj - aj.T @ ai)), 0)


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
