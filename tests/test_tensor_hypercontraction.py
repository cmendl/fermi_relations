import unittest
import numpy as np
import scipy.sparse.linalg as spla
from scipy.stats import ortho_group, unitary_group
from scipy.linalg import expm
import fermi_relations as fr


class TestTensorHypercontraction(unittest.TestCase):
    """
    Test the tensor-hypercontraction form of a molecular Hamiltonian interaction term.

    References:
      - Joonho Lee, Dominic W. Berry, Craig Gidney, William J. Huggins,
        Jarrod R. McClean, Nathan Wiebe, Ryan Babbush
        Even more efficient quantum computations of chemistry through tensor hypercontraction
    	PRX Quantum 2, 030305 (2021)
      - Maxine Luo, J. Ignacio Cirac
        Efficient simulation of quantum chemistry problems in an enlarged basis set
        PRX Quantum 6, 010355 (2025)
      - Yu Wang, Maxine Luo, Matthias Reumann, Christian B. Mendl
        Enhanced Krylov methods for molecular Hamiltonians:
        Reduced memory cost and complexity scaling via tensor hypercontraction
        J. Chem. Theory Comput. 21, 6874 (2025)
    """

    def test_tensor_hypercontraction(self):

        rng = np.random.default_rng()

        # number of modes
        nmodes = 7

        # whether the THC basis transformation matrix is orthogonal
        for ortho in [True, False]:
            # THC rank
            rank = 3 if ortho else 13

            kernel, transform = _random_thc_interaction(nmodes, rank, 0.1, ortho, rng)

            # construct the degree-four interaction coefficient tensor
            vint = np.einsum(kernel,    (4, 5),
                             transform, (0, 4),
                             transform, (1, 4),
                             transform, (2, 5),
                             transform, (3, 5),
                             (0, 1, 2, 3))

            # check symmetries
            self.assertTrue(np.allclose(vint.transpose(1, 0, 2, 3), vint))
            self.assertTrue(np.allclose(vint.transpose(0, 1, 3, 2), vint))
            self.assertTrue(np.allclose(vint.transpose(2, 3, 0, 1), vint))

            vop_ref = fr.molecular_interaction(vint, physics_convention=False)

            # orbital number operators w.r.t. new basis
            nlist = [fr.orbital_number_op(transform[:, i]) for i in range(rank)]
            # define interaction operator using number operators
            vop = 0
            for i in range(rank):
                for j in range(rank):
                    if ortho:
                        # only number operators w.r.t. the same (common) basis commute
                        self.assertAlmostEqual(spla.norm(fr.comm(nlist[i], nlist[j])), 0)
                    vop += 0.5 * kernel[i, j] * (nlist[i] @ nlist[j])
            # compare
            self.assertAlmostEqual(spla.norm(vop - vop_ref), 0, delta=1e-14)

            if ortho:
                # apply number operator interaction terms to a Slater determinant
                # via discrete Hubbard-Stratonovich transformations

                # create a random Slater determinant
                # number of particles
                nptcl = 4
                # random orthonormal states
                base = unitary_group.rvs(nmodes, random_state=rng)
                orb_init = base[:, :nptcl]
                psi = fr.slater_determinant(orb_init)

                t = 0.8

                # reference time-evolved state
                psi_t_ref = expm(-1j * t * vop.todense()) @ psi

                # time evolution using Slater determinants
                coeff_list, orb_list = apply_thc_evolution_slater(kernel, transform, orb_init, t)
                # orbital bases must be isometries
                for orb in orb_list:
                    self.assertTrue(np.allclose(orb.conj().T @ orb, np.identity(orb.shape[1])))
                psi_t = sum(c * fr.slater_determinant(orb) for c, orb in zip(coeff_list, orb_list))
                # compare
                self.assertTrue(np.allclose(psi_t, psi_t_ref))


def _random_thc_interaction(nmodes: int, rank: int, tau: float,
                            ortho_transform: bool, rng: np.random.Generator):
    """
    Generate random tensors according to the tensor hypercontraction representation
    of the interaction term of a molecular Hamiltonian.
    """
    # kernel
    x = rng.standard_normal((rank, rank))
    x = x * np.exp(-tau * np.arange(rank))
    x /= np.linalg.norm(x)
    # ensure that kernel is positive semidefinite
    kernel = x @ x.T

    if ortho_transform:
        # using an isometry as transformation
        assert rank <= nmodes
        transform = ortho_group.rvs(nmodes, random_state=rng)[:, :rank]
    else:
        transform = rng.standard_normal((nmodes, rank))
        for i in range(rank):
            transform[:, i] /= np.linalg.norm(transform[:, i])

    return kernel, transform


def apply_thc_evolution_slater(kernel, transform, orb_init, t: float):
    """
    Time-evolve a Slater determinant governed by the interaction in tensor-hypercontraction form,
    returning the evolved state as a list of Slater determinants.
    """
    assert np.isrealobj(kernel)
    assert np.isrealobj(transform)
    # must be an isometry
    assert np.allclose(transform.T @ transform, np.identity(transform.shape[1]))

    nmodes = transform.shape[0]
    rank = kernel.shape[0]
    assert rank <= nmodes

    # extension to full basis
    u, _ = np.linalg.qr(transform, mode="complete")
    u[:, :rank] = transform
    assert np.allclose(u.T @ u, np.identity(nmodes))

    # represent sum of Slater determinants by coefficients and orbital basis states
    orb_list = [u.T @ orb_init]  # switch to eigenbasis of number operators
    coeff_list = [1]

    for i in range(rank):
        for j in range(rank):
            tau = t * 0.5 * kernel[i, j]
            if i == j:
                # n_i^2 = n_i
                x = np.exp(-1j * tau)
                a = np.identity(nmodes, dtype=complex)
                a[i, i] = x
                orb_list = [a @ orb for orb in orb_list]
            else:
                # note: mu is complex-valued in general
                mu = np.arccos(np.exp(0.5j * tau))
                # absolute values of 'x' and 'y' are not 1 in general since 'mu' is complex
                x = np.exp( 1j * mu - 0.5j * tau)
                y = np.exp(-1j * mu - 0.5j * tau)
                # use Hubbard-Stratonovich transformation
                # to represent time-evolved state as a sum of twice as many Slater determinants
                a = np.identity(nmodes, dtype=complex)
                b = np.identity(nmodes, dtype=complex)
                a[i, i] = x
                a[j, j] = y
                b[i, i] = y
                b[j, j] = x
                orb_list_next = []
                coeff_list_next = []
                for c, orb in zip(coeff_list, orb_list):
                    orb_a, ovl_a = fr.orthonormalize_slater_determinant(a @ orb)
                    orb_b, ovl_b = fr.orthonormalize_slater_determinant(b @ orb)
                    orb_list_next.append(orb_a)
                    orb_list_next.append(orb_b)
                    # phase factor always seems to be real;
                    # could be absorbed into orbitals
                    coeff_list_next.append(0.5 * c * ovl_a.real)
                    coeff_list_next.append(0.5 * c * ovl_b.real)
                orb_list = orb_list_next
                coeff_list = coeff_list_next
    assert len(coeff_list) == len(orb_list)

    # undo base change
    orb_list = [u @ orb for orb in orb_list]

    return coeff_list, orb_list


if __name__ == '__main__':
    unittest.main()
