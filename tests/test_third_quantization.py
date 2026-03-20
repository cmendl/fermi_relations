import unittest
from scipy import sparse
import scipy.sparse.linalg as spla
import fermi_relations as fr


class TestThirdQuantization(unittest.TestCase):
    """
    Test "third quantization" relations.
    """

    def test_anti_comm_ad_a(self):
        """
        Verify that {ai†, aj} == delta_{ij}
        for "third quantization" fermionic operators ai† and aj.
        """
        for nmodes in range(1, 6):
            clist, alist, _ = fr.construct_third_quantization_operators(nmodes)
            for i in range(2*nmodes):
                for j in range(2*nmodes):
                    ci = clist[i]
                    cj = clist[j]
                    ai = alist[i]
                    aj = alist[j]
                    delta = (1 if i == j else 0)
                    self.assertEqual(spla.norm(fr.anti_comm(ci, aj)
                                               - delta * sparse.identity(2**(2*nmodes))), 0)
                    self.assertEqual(spla.norm(fr.anti_comm(ci, cj)), 0)
                    self.assertEqual(spla.norm(fr.anti_comm(ai, aj)), 0)

    def test_comm_n_a(self):
        """
        Verify that [n, a†] == a† and [a, n] == a.
        """
        for nmodes in range(1, 6):
            clist, alist, nlist = fr.construct_third_quantization_operators(nmodes)
            for i in range(2*nmodes):
                c = clist[i]
                a = alist[i]
                n = nlist[i]
                self.assertEqual(spla.norm(fr.comm(n, c) - c), 0)
                self.assertEqual(spla.norm(fr.comm(a, n) - a), 0)


if __name__ == "__main__":
    unittest.main()
