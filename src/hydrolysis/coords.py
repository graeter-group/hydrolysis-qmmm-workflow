import os
import numpy as np
from hydrolysis.parsing import Gro
from hydrolysis.units import *
from hydrolysis.constants import N_QW, QM_WATER_CUTOFF
from math import degrees, sqrt
import logging

logger = logging.getLogger("coords")


def normalize(v):
    return v / np.linalg.norm(v)


def find_qm_waters_and_oh(
    gro: Gro,
    ix_c_carbonyl: int,
    ix_o_carbonyl: int,
    ix_n_peptide: int,
    ix_c_alpha: int,
    qm_water_cutoff: nm = QM_WATER_CUTOFF,
) -> tuple[dict, dict]:

    cwd = os.getcwd()

    c_carbonyl = gro.atoms[ix_c_carbonyl]
    o_carbonyl = gro.atoms[ix_o_carbonyl]
    n_peptide = gro.atoms[ix_n_peptide]
    c_alpha = gro.atoms[ix_c_alpha]

    # find one more than needed because one will be replaced by OH
    n_qm_water_candidates = N_QW + 1
    water_candidates = []
    for i, atom in enumerate(gro.atoms):
        if atom.residue_name != "SOL":
            continue
        if atom.atom_name == "OW":
            d = atom.distance(c_carbonyl)
            if d < qm_water_cutoff:
                water_candidates.append((i, d))

    water_candidates = sorted(water_candidates, key=lambda x: x[1])[
        :n_qm_water_candidates
    ]

    qm_waters = {
        i: {
            "atoms": (gro.atoms[i], gro.atoms[i + 1], gro.atoms[i + 2]),
            "distance": d,
        }
        for i, d in water_candidates
    }

    if len(qm_waters) != n_qm_water_candidates:
        m = f"Only found {len(qm_waters)} qm waters within {qm_water_cutoff}, expected {n_qm_water_candidates}."
        logger.error(m)
        raise ValueError(m)

    for i, water in qm_waters.items():
        o_water = water["atoms"][0]
        h1_water = water["atoms"][1]
        h2_water = water["atoms"][2]
        assert o_water.atom_name == "OW", o_water
        assert h1_water.atom_name == "HW1", h1_water
        assert h2_water.atom_name == "HW2", h2_water
        c = np.array(c_carbonyl.position)
        o = np.array(o_carbonyl.position)
        n = np.array(n_peptide.position)
        ca = np.array(c_alpha.position)
        ow = np.array(o_water.position)
        n_c = c - n
        c_ca = ca - c
        o_c = c - o
        c_ow = ow - c
        plane_normal = np.cross(n_c, c_ca)
        plane_normal = normalize(plane_normal)
        c_ow_projected = c_ow - np.dot(c_ow, plane_normal) * plane_normal
        c_ow_projected = normalize(c_ow_projected)

        # Bürgi-Dunitz angle
        # O-C-O angle close to angle of 107 deg
        # The BD is the angle between the approach vector of O_nucl
        # and the electrophilic C and the C=O bond
        bd = c_carbonyl.angle(left=o_water, right=o_carbonyl)

        bd_penalty = abs(bd - 107)
        qm_waters[i]["bd_penalty"] = bd_penalty

        # Flippin-Lodge angle
        # The FL is an angle that estimates the displacement of the nucleophile,
        # at its elevation, toward or away from the particular R and R' substituents
        # attached to the electrophilic atom
        fl = degrees(np.arccos(np.dot(c_ow_projected, o_c) / (1 * np.linalg.norm(o_c))))
        fl_penalty = abs(fl - 0)
        qm_waters[i]["fl_penalty"] = fl_penalty
        penalty = sqrt(bd_penalty**2 + fl_penalty**2)
        qm_waters[i]["penalty"] = penalty
        qm_waters[i]["ix"] = i

    best_candidate = qm_waters[min(qm_waters, key=lambda x: qm_waters[x]["penalty"])]

    return qm_waters, best_candidate
