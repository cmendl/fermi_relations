from fermi_relations import construct_fermionic_operators


def construct_majorana_operators(nmodes: int):
    """
    Generate sparse matrix representations of the fermionic Majorana operators
    for `nmodes` modes (or sites).
    """
    clist, alist, _ = construct_fermionic_operators(nmodes)
    mlist = [[c + a, 1j*(c - a)] for c, a in zip(clist, alist)]
    return [m for mtuple in mlist for m in mtuple]
