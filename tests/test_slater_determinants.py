"""
Test properties and relations of Slater determinants.
"""

import numpy as np
import fermi_relations as fr


def test_slater_determinant():
    """
    Test Slater determinant construction and basic properties.
    """
    rng = np.random.default_rng()

    # number of modes
    nmodes = 7
    # number of particles
    nptcl = 3
    for orthonormal in [False, True]:
        # create a random Slater determinant
        psi = fr.SlaterDeterminant.random(nmodes, nptcl, orthonormal=orthonormal, rng=rng)
        assert psi.nmodes == nmodes
        assert psi.nptcl  == nptcl
        if orthonormal:
            # must be normalized
            assert abs(psi.norm() - 1) < 1e-13
        # convert to a state vector
        psi_vec_ref = psi.to_vector()
        assert abs(psi.norm() - np.linalg.norm(psi_vec_ref)) < 1e-13
        psi.orthonormalize_orbitals()
        assert abs(abs(psi.coeff) - np.linalg.norm(psi_vec_ref)) < 1e-13
        psi_vec = psi.to_vector()
        assert np.allclose(psi_vec, psi_vec_ref)
        # set coefficient to a random phase factor
        psi.coeff = np.exp(2 * np.pi * 1j * rng.uniform())
        # must be normalized
        assert abs(psi.norm() - 1) < 1e-13
        # must be an eigenstate of the number operator
        psi_vec = psi.to_vector()
        assert np.linalg.norm(fr.total_number_op(nmodes) @ psi_vec - nptcl*psi_vec) < 1e-13
        base = np.linalg.qr(psi.phi, mode="complete")[0]
        for i in range(nmodes):
            # number operator of an individual mode
            n = fr.orbital_number_op(base[:, i])
            assert np.linalg.norm(n @ psi_vec - (1 if i < nptcl else 0) * psi_vec) < 1e-13


def test_vdot_slater():
    """
    Test inner product of two Slater determinants.
    """
    rng = np.random.default_rng()
    # number of modes
    nmodes = 8
    for nptcl0 in range(nmodes + 1):
        for nptcl1 in range(nmodes + 1):
            chi = fr.SlaterDeterminant.random(nmodes, nptcl0, orthonormal=False, rng=rng)
            psi = fr.SlaterDeterminant.random(nmodes, nptcl1, orthonormal=True,  rng=rng)
            # compare with inner product of state vectors (as reference)
            assert abs(fr.vdot_slater(chi, psi)
                    - np.vdot(chi.to_vector(), psi.to_vector())) < 1e-13
