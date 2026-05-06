# TODO:
# HIS is assigned to HIE, no charge, by pdb2gmx
# based on checking H-bonds in the starting structure
# so this may change!
# May need to get charge of the residue dynamically with mda.
# but does this very by triple vs. single?
# residue_charge = qm_atoms.accumulate('charges')
from hydrolysis.units import *

HAMILTON_CACHE = '/data/hamilton/'

AA_CHARGE = {
    "LYS": +1,
    "ARG": +1,
    "ASP": -1,
    "GLU": -1,
    # "HIS": +1,
}

# sentinel files
NO_WATER_FOUND = "no-water-found.info"

# number of QM waters
N_QW = 14

"""
Cutoff for QM waters
NOTE:
Can't be too narrow, otherwise we won't find enough waters.
But if not enough are found within this cutoff
We should raise an error.
Because increasing it could potentially include
QW water separated from the others by the other strands
of the triplehelix.
"""
QM_WATER_CUTOFF = nm(0.8)


TI_TARGET_DISTANCE = 0.145

HBOND = 0.35  # H-bond cutoff in nm

