import logging
import numpy as np
from hydrolysis.constants import N_QW, QM_WATER_CUTOFF
from hydrolysis.parsing import GroAtom, read_gro
import pytest
from math import isclose, sqrt
from hydrolysis.coords import find_qm_waters_and_oh, normalize
from hydrolysis.units import *

logger = logging.getLogger(__name__)


def test_normalize():
    v = np.array([1, 0, 0])
    assert normalize(v).tolist() == [1, 0, 0]

    v = np.array([1, 1, 0])
    assert normalize(v).tolist() == [1 / sqrt(2), 1 / sqrt(2), 0]

    v = np.array([1, 1, 1])
    assert normalize(v).tolist() == [1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)]


def test_distance():
    a = GroAtom(1, "ALA", "CA", 1, (0, 0, 0))
    b = GroAtom(2, "ALA", "CA", 2, (1, 0, 0))
    c = GroAtom(2, "ALA", "CA", 2, (1, 1, 0))
    d = GroAtom(2, "ALA", "CA", 2, (1, 1, 1))

    assert a.distance(b) == 1 == b.distance(a)
    assert b.distance(a) == 1 == a.distance(b)
    assert b.distance(c) == 1 == c.distance(b)
    assert a.distance(d) == sqrt(3) == d.distance(a)


def test_angles_are_invariante_left_right():
    a = GroAtom(1, "ALA", "CA", 1, (0, 0, 0))
    b = GroAtom(2, "ALA", "CA", 2, (1, 0, 0))
    c = GroAtom(3, "ALA", "CA", 3, (0, 1, 0))

    # c
    # a b

    assert a.angle(b, c) == a.angle(c, b) and isclose(a.angle(b, c), 90)
    assert b.angle(a, c) == b.angle(c, a) and isclose(b.angle(a, c), 45)
    assert c.angle(a, b) == c.angle(b, a) and isclose(c.angle(a, b), 45)

    a = GroAtom(1, "ALA", "CA", 1, (0, 0, 0))
    b = GroAtom(2, "ALA", "CA", 2, (0, 1, 0))
    c = GroAtom(3, "ALA", "CA", 3, (0, 0, 1))

    assert a.angle(b, c) == a.angle(c, b) and isclose(a.angle(b, c), 90)
    assert b.angle(a, c) == b.angle(c, a) and isclose(b.angle(a, c), 45)
    assert c.angle(a, b) == c.angle(b, a) and isclose(c.angle(a, b), 45)


def test_angle_warns_on_collisions(caplog):
    a = GroAtom(1, "ALA", "CA", 1, (0, 0, 0))
    b = GroAtom(2, "ALA", "CA", 2, (1, 0, 0))
    c = GroAtom(3, "ALA", "CA", 3, (0, 0, 0))

    assert a.angle(b, c) == 0
    assert "Atoms are colliding" in caplog.text


def test_find_qm_waters_and_oh_by_example():
    gro = read_gro("run_1/chain-a-npt.gro")
    cc = 283
    oc = 284
    n = 285
    ca = 280

    assert gro.atoms[cc].atom_name == "C"
    assert gro.atoms[oc].atom_name == "O"
    assert gro.atoms[n].atom_name == "N"
    assert gro.atoms[ca].atom_name == "CA"

    # some water picked in VMD
    assert isclose(
        gro.atoms[cc].angle(gro.atoms[42528], gro.atoms[oc]), 109.59, abs_tol=0.01
    )
    assert isclose(
        gro.atoms[cc].angle(gro.atoms[101895], gro.atoms[oc]), 151.55, abs_tol=0.01
    )

    qm_waters, best_candidate = find_qm_waters_and_oh(
        gro=gro,
        ix_c_carbonyl=cc,
        ix_o_carbonyl=oc,
        ix_n_peptide=n,
        ix_c_alpha=ca,
        qm_water_cutoff=QM_WATER_CUTOFF,
    )

    assert len(qm_waters) == N_QW + 1
    # at ix 42528
    assert type(best_candidate["atoms"][0]) == GroAtom
    assert best_candidate["atoms"][0].ix == 42528
    assert isclose(best_candidate["bd_penalty"], abs(109.59 - 107), abs_tol=0.01)
    assert isclose(best_candidate["fl_penalty"], 2.53, abs_tol=0.01)
    assert isclose(best_candidate["distance"], 0.391, abs_tol=0.01)

    # candidate is still in qm_waters
    assert qm_waters[best_candidate["ix"]]


def test_find_qm_waters_complains_if_cutoff_too_narrow():
    gro = read_gro("src/tests/test_files/chain-a-npt.gro")
    cc = 283
    oc = 284
    n = 285
    ca = 280

    with pytest.raises(ValueError) as e:
        _, _ = find_qm_waters_and_oh(
            gro=gro,
            ix_c_carbonyl=cc,
            ix_o_carbonyl=oc,
            ix_n_peptide=n,
            ix_c_alpha=ca,
            qm_water_cutoff=nm(0.1),
        )
    assert e.value.args[0] == "Only found 0 qm waters within 0.1, expected 15."

    with pytest.raises(ValueError) as e:
        _, _ = find_qm_waters_and_oh(
            gro=gro,
            ix_c_carbonyl=cc,
            ix_o_carbonyl=oc,
            ix_n_peptide=n,
            ix_c_alpha=ca,
            qm_water_cutoff=nm(0.5),
        )
    assert e.value.args[0] == "Only found 10 qm waters within 0.5, expected 15."


def test_find_enough_qm_waters_in_triplehelix():
    gro = read_gro("src/tests/test_files/triple-npt.gro")
    cc = 283
    oc = 284
    n = 285
    ca = 280

    _, _ = find_qm_waters_and_oh(
        gro=gro,
        ix_c_carbonyl=cc,
        ix_o_carbonyl=oc,
        ix_n_peptide=n,
        ix_c_alpha=ca,
        qm_water_cutoff=QM_WATER_CUTOFF,
    )
