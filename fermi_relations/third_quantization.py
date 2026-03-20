from functools import cache
import numpy as np
from scipy import sparse
from fermi_relations import construct_majorana_operators, construct_majorana_string_basis


@cache
def construct_third_quantization_operators(nmodes: int):
    """
    Construct "third quantization" fermionic operators.

    Reference:
        Tomaž Prosen:
        Third quantization: a general method to solve master equations
        for quadratic open Fermi systems
    	New J. Phys. 10, 043026 (2008)
    """
    mlist  = construct_majorana_operators(nmodes)
    mbasis = construct_majorana_string_basis(nmodes)
    clist = []
    for i in range(2*nmodes):
        c = np.zeros(2 * (2**(2*nmodes),), dtype=complex)
        for jm in range(2**(2*nmodes)):
            # corresponding Majorana string
            jms = mbasis[:, jm].reshape(2 * (2**nmodes,))
            if (jm & (1 << i)) == 0:
                c[:, jm] = mbasis.conj().T @ (mlist[i] @ jms).toarray().reshape(-1) / 2**nmodes
        clist.append(sparse.csr_matrix(c))
    # corresponding annihilation operators
    alist = [sparse.csr_matrix(c.conj().T) for c in clist]
    # corresponding number operators
    nlist = [sparse.csr_matrix(c @ c.conj().T) for c in clist]
    for n in nlist:
        n.eliminate_zeros()
    return clist, alist, nlist
