"""
Slater determinants and related functions.
"""

import numpy as np
from scipy.stats import unitary_group
from fermi_relations.fermiops import orbital_create_op
from fermi_relations.util import crandn


class SlaterDeterminant:
    """
    Slater determinant constructed from single-particle "orbital" states.
    """
    def __init__(self, phi, coeff: float | complex = 1.0):
        # store the single-particle states as column vectors in `phi`
        self.phi = np.asarray(phi)
        self.coeff = coeff

    @property
    def nptcl(self) -> int:
        """
        Number of particles in the Slater determinant.
        """
        return int(self.phi.shape[1])

    @classmethod
    def random(cls, nmodes: int, nptcl: int, orthonormal: bool = True,
               rng: np.random.Generator | None = None):
        """
        Generate a random Slater determinant with `nptcl` particles in `nmodes` modes.
        """
        if rng is None:
            rng = np.random.default_rng()
        # pylint: disable=no-else-return
        if orthonormal:
            # random orthonormal states
            base = unitary_group.rvs(nmodes, random_state=rng)
            orb = base[:, :nptcl]
            phase = np.exp(2 * np.pi * 1j * rng.uniform())
            return cls(orb, phase)
        else:
            # random generic states
            return cls(0.5 * crandn((nmodes, nptcl), rng), crandn(rng=rng))

    def orthonormalize_states(self):
        """
        In-place orthonormalize the orbital states defining the Slater determinant,
        and absorb the normalization factor into the coefficient.
        """
        q, _ = np.linalg.qr(self.phi, mode="reduced")
        self.coeff *= np.linalg.det(q.conj().T @ self.phi).item()  # convert to native Python scalar
        self.phi = q
        # enable chaining
        return self

    def norm(self):
        """
        Norm of the Slater determinant.
        """
        return abs(self.coeff) * np.sqrt(abs(np.linalg.det(self.phi.conj().T @ self.phi)))

    def __rmul__(self, factor: float | complex):
        """
        Logical scaling of the Slater determinant by `factor`.
        """
        return SlaterDeterminant(self.phi, factor * self.coeff)

    def transform_by(self, m):
        """
        Multiply the orbital states by the matrix `m`.

        Note that this operation is not equivalent to
        a linear transformation on the whole Fock space.
        """
        return SlaterDeterminant(m @ self.phi, self.coeff)

    def to_vector(self):
        """
        Construct the state vector representation of the Slater determinant on the whole Fock space.
        """
        nmodes = self.phi.shape[0]
        # vacuum state
        psi = np.zeros(2**nmodes)
        psi[0] = 1
        for i in reversed(range(self.phi.shape[1])):
            psi = orbital_create_op(self.phi[:, i]) @ psi
        return self.coeff * psi


def vdot_slater(chi: SlaterDeterminant, psi: SlaterDeterminant):
    """
    Inner product `<chi | phi>` of two Slater determinants.
    """
    if chi.nptcl != psi.nptcl:
        return 0.
    return chi.coeff.conjugate() * psi.coeff * np.linalg.det(chi.phi.conj().T @ psi.phi)
