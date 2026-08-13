"""
Test spin and charge pseudospin SU(2) symmetries of the Fermi-Hubbard Hamiltonian.
"""

from scipy import sparse
import scipy.sparse.linalg as spla
import fermi_relations as fr


def test_fermi_hubbard_spin_comm():
    """
    Probe the spin SU(2) symmetry of the Fermi-Hubbard Hamiltonian.
    """
    # Hamiltonian parameters
    t  =  1.3
    u  =  3.2
    mu = -0.7
    for nsites in range(1, 6):
        sup, sdn, sz = construct_spin_operators(nsites)
        # spin commutation relations
        assert spla.norm(fr.comm(sz,  sup) - sup)  == 0
        assert spla.norm(fr.comm(sz,  sdn) + sdn)  == 0
        assert spla.norm(fr.comm(sup, sdn) - 2*sz) == 0
        sx =  0.5  * (sup + sdn)
        sy = -0.5j * (sup - sdn)
        assert spla.norm(fr.comm(sx, sy) - 1j * sz) == 0
        assert spla.norm(fr.comm(sy, sz) - 1j * sx) == 0
        assert spla.norm(fr.comm(sz, sx) - 1j * sy) == 0
        # commutation relations between the spin operators and the Hamiltonian
        hmat = construct_fermi_hubbard_1d_hamiltonian(nsites, t, u, mu)
        assert spla.norm(fr.comm(hmat, sz))  < 1e-13
        assert spla.norm(fr.comm(hmat, sup)) < 1e-13
        assert spla.norm(fr.comm(hmat, sdn)) < 1e-13


def test_fermi_hubbard_pseudospin_comm():
    """
    Probe the charge pseudospin SU(2) symmetry of the Fermi-Hubbard Hamiltonian.
    """
    # Hamiltonian parameters
    t  = 1.1
    u  = 2.7
    mu = 0.4
    for nsites in range(1, 6):
        tup, tdn, tz = construct_pseudospin_operators(nsites)
        # spin commutation relations
        assert spla.norm(fr.comm(tz,  tup) - tup)  == 0
        assert spla.norm(fr.comm(tz,  tdn) + tdn)  == 0
        assert spla.norm(fr.comm(tup, tdn) - 2*tz) == 0
        tx =  0.5  * (tup + tdn)
        ty = -0.5j * (tup - tdn)
        assert spla.norm(fr.comm(tx, ty) - 1j * tz) == 0
        assert spla.norm(fr.comm(ty, tz) - 1j * tx) == 0
        assert spla.norm(fr.comm(tz, tx) - 1j * ty) == 0
        # commutation relations between the pseudospin operators and the Hamiltonian
        hmat = construct_fermi_hubbard_1d_hamiltonian(nsites, t, u, mu)
        assert spla.norm(fr.comm(hmat, tz)) < 1e-13
        assert spla.norm(fr.comm(hmat, tup) + 2*mu*tup) < 1e-13
        assert spla.norm(fr.comm(hmat, tdn) - 2*mu*tdn) < 1e-13


def construct_fermi_hubbard_1d_hamiltonian(nsites: int, t: float, u: float, mu: float):
    """
    Construct the Fermi-Hubbard Hamiltonian
    with nearest-neighbor hopping on a one-dimensional lattice as sparse matrix.
    """
    clist, alist, nlist = fr.construct_fermionic_operators(2*nsites)
    # kinetic hopping terms and
    # interaction u (n_up - 1/2) (n_dn - 1/2) and number operator - mu (n_up + n_dn)
    hamiltonian = sum(-t * (clist[j] @ alist[j+2] + clist[j+2] @ alist[j])
                      for j in range(2*nsites - 2)) \
                + sum(u * (nlist[j]   - 0.5*sparse.identity(4**nsites)) \
                        @ (nlist[j+1] - 0.5*sparse.identity(4**nsites)) \
                      - mu * (nlist[j] + nlist[j+1])
                      for j in range(0, 2*nsites, 2))
    return hamiltonian


def construct_spin_operators(nsites: int):
    """
    Construct the spin operators.
    """
    clist, alist, nlist = fr.construct_fermionic_operators(2*nsites)
    sup = sum(clist[2*j] @ alist[2*j+1] for j in range(nsites))
    sdn = sup.conj().T
    sz = 0.5 * sum(nlist[2*j] - nlist[2*j+1] for j in range(nsites))
    return sup, sdn, sz


def construct_pseudospin_operators(nsites: int):
    """
    Construct the pseudospin operators.
    """
    clist, _, nlist = fr.construct_fermionic_operators(2*nsites)
    tup = sum((-1)**j * (clist[2*j] @ clist[2*j+1]) for j in range(nsites))
    tdn = tup.conj().T
    numop = sum(nlist[j] for j in range(2*nsites))
    tz = 0.5 * (numop - nsites*sparse.identity(4**nsites))
    return tup, tdn, tz
