from fermi_relations import construct_fermionic_operators
import numpy as np
from scipy import sparse


def construct_majorana_operators(nmodes: int):
    """
    Generate sparse matrix representations of the fermionic Majorana operators
    for `nmodes` modes (or sites).
    """
    clist, alist, _ = construct_fermionic_operators(nmodes)
    mlist = [[c + a, 1j*(c - a)] for c, a in zip(clist, alist)]
    return [m for mtuple in mlist for m in mtuple]


def orbital_majorana_op(x):
    """
    "Orbital" Majorana operator (linear combination of Majorana operators with coefficients 'x').
    """
    x = np.asarray(x)
    nmodes = len(x) // 2
    mlist = construct_majorana_operators(nmodes)
    return sum(x[i] * mlist[i] for i in range(2*nmodes))


def kinetic_exponential_majorana(nmodes: int, i: int, j: int, t: float):
    """
    Construct the unitary matrix exponential of the kinetic hopping term
    based on Majorana operators.
    """
    mlist = construct_majorana_operators(nmodes)
    numop_proj = 0.5 * (sparse.identity(2**nmodes)
                        + mlist[2*i] @ mlist[2*i+1] @ mlist[2*j] @ mlist[2*j+1])
    itkin = -0.5 * (mlist[2*i] @ mlist[2*j+1] + mlist[2*j] @ mlist[2*i+1])
    # Euler representation
    return sparse.identity(2**nmodes) + (np.cos(t) - 1) * numop_proj - np.sin(t) * itkin


def hubbard_interaction_exponential_majorana(nmodes: int, i: int, j: int, t: float):
    """
    Construct the unitary matrix exponential of
    the Hubbard model interaction term (n_i - 1/2) (n_j - 1/2)
    based on Majorana operators.
    """
    mlist = construct_majorana_operators(nmodes)
    vint = -mlist[2*i] @ mlist[2*i+1] @ mlist[2*j] @ mlist[2*j+1]
    # Euler representation
    return np.cos(0.25*t) * sparse.identity(2**nmodes) - 1j * np.sin(0.25*t) * vint


def construct_majorana_string_basis(nmodes: int):
    """
    Construct an operator basis of Majorana strings, stored as columns in a matrix.
    """
    mlist = construct_majorana_operators(nmodes)
    pmat = sparse.lil_matrix((4**nmodes, 4**nmodes), dtype=complex)
    for m in range(2**(2*nmodes)):
        p = np.identity(2**nmodes, dtype=complex)
        for i in range(2*nmodes):
            if m & (1 << i):
                p = p @ mlist[i]
        # include a potential imaginary unit factor to make the Majorana string Hermitian
        wm = m.bit_count()
        if wm % 4 != 0 and (wm - 1) % 4 != 0:
            p *= 1j
        pmat[m] = p.reshape(-1)
    return sparse.csr_matrix(pmat.T)


def majorana_strings_commute(ima: int, imb: int) -> bool:
    """
    Test whether two Majorana strings (encoded as integers) commute.
    """
    parity = ima.bit_count() * imb.bit_count() - (ima & imb).bit_count()
    return (parity % 2) == 0


def majorana_string_product(ima: int, imb: int) -> tuple[int, complex]:
    """
    Compute the product of two Hermitian Majorana strings encoded as integers,
    and return the resulting Majorana string encoded as an integer and its phase factor.
    """
    # encoded product results from bitwise XOR
    # since the square of any Majorana operator is the identity
    imp = ima ^ imb
    # compute phase factor
    phase = 1
    # sign factor from permutations of individual Majorana operators
    for ia in range(ima.bit_length()):
        if ima & (1 << ia):
            for ib in range(imb.bit_length()):
                if imb & (1 << ib):
                    if ia > ib:
                        phase = -phase
    # phase factors from Hermitian property
    wa = ima.bit_count()
    if wa % 4 != 0 and (wa - 1) % 4 != 0:
        phase *= 1j
    wb = imb.bit_count()
    if wb % 4 != 0 and (wb - 1) % 4 != 0:
        phase *= 1j
    wp = imp.bit_count()
    if wp % 4 != 0 and (wp - 1) % 4 != 0:
        phase *= -1j
    return imp, phase
