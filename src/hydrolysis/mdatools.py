import logging

import MDAnalysis as mda

from hydrolysis.constants import AA_CHARGE

logger = logging.getLogger("mdatools")


def find_offset_chain(u, idx_c_carbonyl: int):
    aces = u.select_atoms("resname ACE")
    idxs = aces.ix
    idxs.sort()
    chain_starts = []
    previous = 0
    for ix in idxs:
        delta = ix - previous
        previous = ix
        if delta > 1:
            chain_starts.append(ix)

    start_b, start_c = chain_starts

    offset = 0
    if idx_c_carbonyl < start_b:
        offset = 0
        chain = "a"
    elif idx_c_carbonyl >= start_b and idx_c_carbonyl < start_c:
        offset = 561
        chain = "b"
    elif idx_c_carbonyl >= start_c:
        offset = 561 + 568
        chain = "c"
    else:
        raise ValueError("This residue index appears nowhere in single chains.")

    return offset, chain


def get_initial_qm_atoms(env, system, gro):
    """
    Writes to single/triple (=system) directory
    """
    ix_c_carbonyl = int(env[f"{system}_ix_c_carbonyl"])
    ix_n_peptide = int(env[f"{system}_ix_n_peptide"])
    u = mda.Universe(gro)
    qm_atoms = u.select_atoms(
        f"same residue as index {ix_c_carbonyl} or same residue as index {ix_n_peptide}"
    )
    o_carbonyl = u.select_atoms(f"name O and same residue as index {ix_c_carbonyl}")[0]
    c_alpha = u.select_atoms(f"name CA and same residue as index {ix_c_carbonyl}")[0]
    ixs_qm = [str(index) for index in qm_atoms.ix]

    charge = 0
    for aa in set(qm_atoms.resnames):
        d_charge = AA_CHARGE.get(aa)
        if d_charge is not None:
            charge += d_charge

    env["charge"] = charge
    env["ixs_qm"] = ixs_qm
    env["ix_o_carbonyl"] = str(o_carbonyl.ix)
    env["ix_c_alpha"] = str(c_alpha.ix)
    return env
