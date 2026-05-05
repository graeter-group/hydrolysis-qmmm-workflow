"""
Operations to setup and run QM/MM simulations.
State and environment is taken from env.json files in the cwd.
Dispatch is to the cluster by default.
"""

import logging
import os
import signal
import subprocess as sp
import time
from pathlib import Path

import MDAnalysis as mda
import pandas as pd
from kimmdy.parsing import read_top, write_top
from kimmdy.topology.utils import get_top_section, set_top_section
from tqdm import tqdm

from src.parsing import read_gro, write_gro
import src.utils as ut
from src.constants import N_QW, QM_WATER_CUTOFF, TI_TARGET_DISTANCE
from src.settings import ASSETS
from src.mdatools import find_offset_chain
from src.coords import find_qm_waters_and_oh
from src.units import A
from src.utils import (
    Result,
    cp2k_reference_to_inp,
    fill_template,
    get_rotation_restraints,
    link,
    log_skipped,
    log_waiting,
    pushd,
    read_env,
    read_xvg,
    safely,
    sh,
    slurm_dispatch,
    unsafe,
    write,
    write_env,
)

logger = logging.getLogger("qm")


def setup_chain(chain: str, force: bool = False) -> Result:
    """
    pdb2gmx; bo; solvate; neutralize 0.15 molar NaCal;
    minimize; nvt; npt; make_ndx
    """
    logger.info(f"Setup system: {chain}.")
    target = f"{chain}-npt.gro"
    if Path(target).exists() and not force:
        log_skipped(target, "setup system")
        return Result.SKIP

    sh(
        f"gmx pdb2gmx -f {chain}.pdb -water tip3p -ff amber99sb-star-ildnp -o {chain}-initial.gro -p {chain}-initial.top -i {chain}-posre.itp -merge all"
    )
    sh(f"cp {chain}-initial.top {chain}-topol.top")
    sh(f"gmx editconf -f {chain}-initial.gro -o {chain}-box.gro -c -d 2.0 ")
    with open(f"{chain}-box.gro", "r") as f:
        l = ""
        # got to the last line, the box vector
        for l in f:
            pass
        boxsize = " ".join([str(float(x) * 1.3) for x in l.split()])
    sh(f"gmx editconf -f {chain}-initial.gro -o {chain}-box.gro -c -box {boxsize}")
    sh(f"gmx solvate -cp {chain}-box.gro -o {chain}-solvated.gro -p {chain}-topol.top ")
    sh(
        f"gmx grompp -f ions.mdp -c {chain}-solvated.gro -p {chain}-topol.top -o {chain}-ions.tpr "
    )
    sh(
        f'echo "SOL" | gmx genion -s {chain}-ions.tpr -o {chain}-neutral.gro -p {chain}-topol.top -pname NA -nname CL -neutral -conc 0.15 '
    )
    sh(
        f"gmx grompp -f minim.mdp -c {chain}-neutral.gro -r {chain}-neutral.gro -p {chain}-topol.top -o {chain}-min.tpr "
    )
    sh(f"gmx mdrun -v -deffnm {chain}-min ")
    sh(
        f"gmx grompp -f nvt.mdp -c {chain}-min.gro -r {chain}-min.gro -p {chain}-topol.top -o {chain}-nvt.tpr "
    )
    sh(f"gmx mdrun -deffnm {chain}-nvt ")
    sh(
        f"gmx grompp -f npt.mdp -c {chain}-nvt.gro -r {chain}-nvt.gro -t {chain}-nvt.cpt -p {chain}-topol.top -o {chain}-npt.tpr"
    )
    sh(f"gmx mdrun -deffnm {chain}-npt")

    return Result.OK


def make_ndx(gro: str, ndx: str) -> Result:
    logger.info("Writing ndx")
    sh(f'echo "q\n" | gmx make_ndx -f {gro} -o {ndx}', quiet=0)
    return Result.OK


def add_terminal_index_groups(system: str, ndx: str) -> Result:
    path_system = Path(system)
    path_index = Path(ndx)
    if not path_system.exists():
        raise FileNotFoundError(path_system)

    if not path_index.exists():
        raise FileNotFoundError(path_index)

    u = mda.Universe(path_system)
    nmes = u.select_atoms("resname NME and name CH3")
    aces = u.select_atoms("resname ACE and name CH3")

    with open(path_index, "a") as f:
        f.write("[ CTERM ]\n")
        f.writelines([str(i + 1) + "\n" for i in nmes.indices])
        f.write("[ NTERM ]\n")
        f.writelines([str(i + 1) + "\n" for i in aces.indices])

    return Result.OK


def equilibrate_chain(chain: str, local=False, force=False) -> Result:
    env = read_env()
    cwd = os.getcwd()
    if not env:
        logger.error(
            f"Could not load env.json for op.equilibrate_chain in {cwd}. Skipping."
        )
        return Result.WAIT
    if chain == "triple":
        system = "triple"
    else:
        system = "single"
    external_force = env[f"{system}_external_force"]
    n_steps_eq = env["n_steps_eq"]

    logger.info(f"Equilibrate {chain} under external force: {external_force}")
    job = f"{chain}-eq"
    target = f"{job}.gro"
    logger.info(f"Equilibrate system {chain} under external force {external_force}.")

    if Path(target).exists() and not force:
        log_skipped(target, f"Already equilibrated {chain} in {cwd}")
        return Result.SKIP

    target_tpr = f"{job}.tpr"
    if Path(target_tpr).exists() and not force:
        log_skipped(target_tpr, f"Already created tpr for {chain} in {cwd}.")
    else:
        template = f"{ASSETS}/equilibrate.template.mdp"
        gro = f"{chain}-npt.gro"
        ndx = f"{chain}.ndx"
        top = f"{chain}-topol.top"

        sh(f"cp ../{gro} ../{top} .")

        make_ndx(gro, ndx)
        add_terminal_index_groups(gro, ndx)
        rotation_restraints = get_rotation_restraints(ASSETS, chain)

        fill_template(
            Path(template),
            Path(f"{job}.mdp"),
            EXTERNAL_FORCE=external_force,
            NSTEPS=n_steps_eq,
            ROTATION_RESTRAINTS=rotation_restraints,
        )
        sh(
            f"gmx grompp -n {ndx} -f {job}.mdp -c {gro} -r {gro} -p {top} -o {job}.tpr -maxwarn 1"
        )
    if local:
        sh(f"./local-job.sh {job}")
    else:
        slurm_dispatch(cwd=".", job=job)

    return Result.OK


def analyse_equilibrated_chain(chain: str, force=False) -> Result:
    """Analyse equilibrated chain.

    Args:
        chain: any of "a", "b", "c", "triple"
    """
    job = f"{chain}-eq"
    cwd = os.getcwd()
    trj = f"{job}.xtc"
    target_gro = f"{job}-center.gro"
    target_xtc = f"{job}-center.xtc"
    target_energy = f"{job}-energy.xvg"
    if not Path(trj).exists():
        log_waiting(trj, f"center xtc in {cwd}")
        return Result.WAIT
    if not Path(target_energy).exists() and not force:
        sh(
            f"echo '1 11 12 13 14 16 18 23 24' | gmx energy -f {job}.edr -o {target_energy}",
            quiet=2,
        )
    if Path(target_gro).exists() and Path(target_xtc).exists() and not force:
        log_skipped(f"{target_gro} and {target_xtc}", f"center xtc in {cwd}")
        return Result.SKIP
    logger.info(f"Centering xtcs for {job} in {cwd}")
    sh(
        f"echo 'Protein\n non-Water' | gmx trjconv -f {trj} -s {job}.tpr -pbc mol -dump -1 -center -o {target_gro}",
        quiet=2,
    )
    sh(
        f"echo 'Protein\n non-Water' | gmx trjconv -f {trj} -s {job}.tpr -pbc mol -center -o {target_xtc}",
        quiet=2,
    )

    return Result.OK


def export_equilibrated_chain_frames(chain: str, n: int, force: bool = False) -> Result:
    """
    Needs equilibrated systems.
    Export n final frames from the equilibration
    to search for suitable starting water configurations.
    """
    env = read_env()
    cwd = os.getcwd()
    if not env:
        logger.error(
            f"Could not load env.json for op.equilibrate_chain in {cwd}. Skipping."
        )
        return Result.WAIT
    logger.info(f"Export frames for {cwd}, chain {chain}")
    dt = 0.002  # ps
    n_steps_eq = int(env["n_steps_eq"])
    last_time_ps = n_steps_eq * dt
    nstxtcout = 500  # every 1 ps = 5000 ps / frame
    step_ps = nstxtcout * dt  # ps
    first_time_ps = last_time_ps - n * step_ps
    frames_ps = range(int(first_time_ps) + 1, int(last_time_ps) + 1, int(step_ps))

    eq_gro = f"{chain}-eq.gro"
    if not Path(eq_gro).exists():
        log_waiting(eq_gro, f"extract_frames in {cwd}")
        return Result.WAIT
    target_dir = Path(f"{chain}-frames/")
    if target_dir.is_dir() and target_dir.exists() and not force:
        log_skipped(target_dir, f"extract_frames in {cwd}")
        return Result.SKIP
    if not target_dir.exists():
        target_dir.mkdir()
    for frame in tqdm(frames_ps):
        sh(
            f"echo 'System' | gmx trjconv -f {chain}-eq.xtc -s {chain}-eq.tpr -dump {frame} -o {target_dir}/{chain}-frame-{frame}.gro",
            quiet=2,
        )

    return Result.OK


def match_triple_to_single_chain_residues(force: bool = False) -> Result:
    """
    Writes it's result to the peptide_bonde_dir env.json.
    """
    logger.info("Matching ixs of peptide bond from triplehelix to single chain")
    env = read_env()
    cwd = os.getcwd()
    if not env:
        logger.error(
            "Could not load env.json for op.match_triple_to_single_chain_residues. Skipping."
        )
        raise AssertionError("missing step")
    if env.get("chain") and not force:
        logger.info(f"Found existing chain in environment in {cwd}.")
        return Result.SKIP
    triple_ix_c_carbonyl = int(env["triple_ix_c_carbonyl"])
    triple_ix_n_peptide = int(env["triple_ix_n_peptide"])
    u = mda.Universe("../triple-eq.gro")
    offset, chain = find_offset_chain(u, triple_ix_c_carbonyl)
    env["chain"] = chain
    env["single_ix_c_carbonyl"] = str(triple_ix_c_carbonyl - offset)
    env["single_ix_n_peptide"] = str(triple_ix_n_peptide - offset)
    write_env(env)

    return Result.OK


def create_wethyd_gro(parent_gro: Path | str, out_gro: Path | str, force=True):
    cwd = os.getcwd()
    env = read_env()
    if not env:
        logger.error(
            f"Could not load env.json for op.equilibrate_chain in {cwd}. Skipping."
        )
        return Result.WAIT
    gro = read_gro(parent_gro)
    if not gro:
        logger.error(
            f"Could not load gro file {parent_gro} for op.create_wethyd_gro in {cwd}. Skipping."
        )
        return Result.WAIT

    ix_c_carbonyl = int(env[f"ix_c_carbonyl"])
    ix_o_carbonyl = int(env[f"ix_o_carbonyl"])
    ix_c_alpha = int(env[f"ix_c_alpha"])
    ix_n_peptide = int(env[f"ix_n_peptide"])

    qm_waters, best_candidate = find_qm_waters_and_oh(
        gro=gro,
        ix_c_carbonyl=ix_c_carbonyl,
        ix_o_carbonyl=ix_o_carbonyl,
        ix_c_alpha=ix_c_alpha,
        ix_n_peptide=ix_n_peptide,
    )
    logger.info(f"Best candidate for OH: {best_candidate}")

    # NOTE: never trust the atom numbers of a gro file!
    ix_ow, ix_hw1, ix_hw2 = [a.ix for a in best_candidate["atoms"]]

    # turn into hydroxide
    ow = gro.atoms[ix_ow]
    hw1 = gro.atoms[ix_hw1]
    hw2 = gro.atoms[ix_hw2]
    ow.residue_name = "OH"

    ow.atom_name = "O1"
    hw1.residue_name = "OH"
    hw1.atom_name = "H1"

    # remove oh from the regular qm waters
    qm_waters.pop(ix_ow)

    water_to_qm_water_atomname = {
        "OW": "QOW",
        "HW1": "QHW1",
        "HW2": "QHW2",
    }
    for i, water in qm_waters.items():
        atoms = water["atoms"]
        for atom in atoms:
            atom.residue_name = "QW"
            try:
                atom.atom_name = water_to_qm_water_atomname[atom.atom_name]
            except KeyError:
                raise KeyError(f"something has gone wrong with {atom}")

    # first remove last CL, then remove water (and place as OH at the end)
    # find last CL
    ix_cl = None
    for i, atom in enumerate(gro.atoms):
        if atom.residue_name == "CL":
            ix_cl = i

    assert ix_cl is not None, "no CL found in gro file"

    # modify the gro file
    ix_to_remove = [ix_ow, ix_hw1, ix_hw2, ix_cl] + [
        x for i in qm_waters.keys() for x in [i, i + 1, i + 2]
    ]

    # remove in reverse order to not mess up the indices
    for i in sorted(ix_to_remove, reverse=True):
        gro.atoms.pop(i)

    # add qm_waters to the gro at the end
    ixs_qw = []
    for i, water in qm_waters.items():
        for atom in water["atoms"]:
            # no point modifying the atom_number
            # because it's capped at 5 digits
            # but we track the index (=line number in the gro file - 3)
            atom.ix = len(gro.atoms)
            gro.atoms.append(atom)
            ixs_qw.append(atom.ix)

    assert (
        len(ixs_qw) == N_QW * 3
    ), f"Expected {N_QW * 3} qm waters, found {len(ixs_qw)}"

    # add OH
    for atom in [ow, hw1]:
        atom.ix = len(gro.atoms)
        gro.atoms.append(atom)

    # did we remove the right number of atoms?
    # removed 1 CL
    # removed one H ot turn H2O into OH
    assert (
        gro.n_atoms == len(gro.atoms) + 2
    ), f"n_atoms was: {gro.n_atoms} and is now {len(gro.atoms)} but should be {gro.n_atoms - 2}"

    # set new number of atoms
    gro.n_atoms = len(gro.atoms)

    logger.info(f"Found ix_o_oh: {ow.ix}")
    env["ix_o_oh"] = str(ow.ix)
    env["ixs_oh"] = [str(x) for x in [ow.ix, hw1.ix]]
    env["ixs_qw"] = [str(x) for x in ixs_qw]
    env["ixs_qm_hyd"] = env["ixs_qm"] + env["ixs_oh"]
    env["ixs_qm_wethyd"] = env["ixs_qm"] + env["ixs_oh"] + env["ixs_qw"]
    env["n_qm_water"] = str(N_QW)  # OH is not counted
    write_env(env)
    write_gro(gro, out_gro)
    return Result.OK


def create_wethyd_top(parent_top: Path, out_top: Path, force: bool = False) -> Result:
    env = read_env()
    if env is None:
        log_waiting("create_wethyd_gro", "create_wethyd_top")
        return Result.WAIT

    if out_top.exists() and not force:
        return Result.SKIP

    n_qm_water = int(env["n_qm_water"])  # without OH
    ixs_qmatoms = env["ixs_qm"]

    top = read_top(parent_top)

    protein_moleculetype = [
        x for x in top.keys() if "Protein" in x and "moleculetype_" in x
    ][0].removeprefix("moleculetype_")

    # disable bonds between protein QM atoms
    newbonds = []
    for a, b, f in [
        bs
        for bs in unsafe(get_top_section, top, "bonds", protein_moleculetype)
        if bs[0] != ";"
    ]:
        if (a in ixs_qmatoms) and (b in ixs_qmatoms):
            new = [a, b, "5"]
        else:
            new = [a, b, f]
        newbonds.append(new)
    set_top_section(top, "bonds", newbonds, protein_moleculetype)
    qw_moleculetype_name = f"moleculetype_QW"

    # qm water does not need bond and angle parameters etc. as they are in QM
    qw_moleculetype = {
        "content": [],
        "else_content": [],
        "extra": [],
        "condition": None,
        "subsections": {},
    }
    qw_moleculetype["content"] = [["QW", "2"]]
    qw_moleculetype["subsections"]["atoms"] = {
        "else_content": [],
        "extra": [],
        "condition": None,
        "content": [
            # id  at type     res nr  res name  at name  cg nr  charge    mass
            ["1", "OW", "1", "QW", "QOW", "1", "-0.834", "16.00000"],
            ["2", "HW", "1", "QW", "QHW1", "1", "0.417", "1.00800"],
            ["3", "HW", "1", "QW", "QHW2", "1", "0.417", "1.00800"],
        ],
    }
    top[qw_moleculetype_name] = qw_moleculetype

    newmolecules = []
    for mol in unsafe(get_top_section, top, "molecules"):
        if mol[0] == "CL":
            mol[1] = str(int(mol[1]) - 1)  # remove on CL to keep neutral charge
        if mol[0] == "SOL":
            mol[1] = str(
                int(mol[1]) - (n_qm_water + 1)
            )  # because in addition to qm waters, one SOL is replaced by OH
        newmolecules.append(mol)

    newmolecules.append(["QW", str(n_qm_water)])
    newmolecules.append(["OH", "1"])
    set_top_section(top, "molecules", newmolecules)

    write_top(top, out_top)

    return Result.OK


def create_wethyd_top_and_gro(
    parent_top: Path,
    parent_gro: Path,
    out_top: Path,
    out_gro: Path,
    force: bool = False,
) -> Result:
    """
    create qm topol and gro files from eq gro and topol files
    Note: order is important, need to put the new OH after all SOL
    https://manual.gromacs.org/documentation/2018/user-guide/run-time-errors.html#xxx-non-matching-atom-names
    contents of your [ molecules ] directive matches the exact order of the atoms in the coordinate file
    """
    logger.info("Generating qm top and gro.")
    gro_result = create_wethyd_gro(parent_gro=parent_gro, out_gro=out_gro, force=force)
    top_result = create_wethyd_top(parent_top=parent_top, out_top=out_top, force=force)
    if gro_result is Result.OK and top_result is Result.OK:
        return Result.OK
    else:
        return Result.WARN


def setup_wethyd_warmup(force: bool = False) -> Result:
    job = "wethyd"
    warmupjob = f"{job}-warmup"
    tpr = f"{warmupjob}.tpr"
    env = read_env()
    cwd = os.getcwd()
    system = cwd.split("/")[-2]
    if not env:
        logger.warning(
            f"Could not load env.json for op.setup_wethyd in {cwd}. Skipping."
        )
        return Result.WARN

    if Path(tpr).exists() and not force:
        logger.info(f"Found existing {tpr} in {cwd}. Skipping.")
        return Result.SKIP

    start_gro = f"{job}-start.gro"
    top = f"{job}.top"

    dep = "eq"
    parent_gro = f"{dep}.gro"
    parent_top = f"{dep}.top"
    create_wethyd_top_and_gro(
        parent_top=Path(parent_top),
        parent_gro=Path(parent_gro),
        out_top=Path(top),
        out_gro=Path(start_gro),
        force=force,
    )

    write_qm_index(job, force)

    tpr = f"{warmupjob}.tpr"
    if Path(tpr).exists() and not force:
        logger.info(f"Found existing {tpr} in {cwd}. Skipping.")
        return Result.WARN

    charge = str(int(env[f"charge"]) - 1)

    generate_qm_reference(charge=charge, job=job, force=force)

    basis_set = env.get("basis_set", "DZVP-MOLOPT-GTH")
    xc_functional = env.get("xc_functional", "PBE")
    cp2k_inp = f"{job}.inp"

    cp2k_reference_to_inp(
        in_reference=f"{job}-qm-reference_cp2k.inp",
        in_template=None,
        out_cp2k_inp=cp2k_inp,
        job=job,
        basis_set=basis_set,
        xc_functional=xc_functional,
    )
    external_force = env[f"{system}_external_force"]
    rotation_restraints = get_rotation_restraints(ASSETS, system)
    fill_template(
        Path(f"{ASSETS}/{warmupjob}.template.mdp"),
        Path(f"{warmupjob}.mdp"),
        CHARGE=charge,
        EXTERNAL_FORCE=external_force,
        ROTATION_RESTRAINTS=rotation_restraints,
        PREFIX=warmupjob,
    )

    sh(
        f"gmx grompp -n {job}.ndx -f {warmupjob}.mdp -c {start_gro} -r {start_gro} -qmi {cp2k_inp} -p {job}.top -o {tpr} -maxwarn 2",
        quiet=1,
    )

    return Result.OK


def setup_wethyd_after_warmup(force: bool = False) -> Result:
    logger.info(f"Setting up wethyd after warmup.")
    job = "wethyd"
    warmupjob = f"{job}-warmup"
    tpr = f"{job}.tpr"
    env = read_env()
    cwd = os.getcwd()
    system = cwd.split("/")[-2]
    if not env:
        logger.warning(
            f"Could not load env.json for op.setup_wethyd in {cwd}. Skipping."
        )
        return Result.WARN
    if Path(tpr).exists() and not force:
        logger.info(f"Found existing {tpr} in {cwd}. Skipping.")
        return Result.SKIP
    gro = f"{warmupjob}.gro"
    if not Path(gro).exists():
        logger.info(f"Found no gro from warmup {gro} in {cwd}. Skipping.")
        return Result.SKIP

    external_force = env[f"{system}_external_force"]
    charge = str(int(env[f"charge"]) - 1)
    rotation_restraints = get_rotation_restraints(ASSETS, system)
    cp2k_inp = f"{job}.inp"  # already exists from warmup
    fill_template(
        Path(f"{ASSETS}/{job}.template.mdp"),
        Path(f"{job}.mdp"),
        CHARGE=charge,
        EXTERNAL_FORCE=external_force,
        ROTATION_RESTRAINTS=rotation_restraints,
        PREFIX=job,
    )

    sh(
        f"gmx grompp -n {job}.ndx -f {job}.mdp -c {gro} -r {gro} -qmi {cp2k_inp} -p {job}.top -o {tpr} -maxwarn 2",
        quiet=1,
    )

    return Result.OK


def setup_wethyd(force: bool = False) -> Result:
    job = "wethyd"
    tpr = f"{job}.tpr"
    env = read_env()
    cwd = os.getcwd()
    system = cwd.split("/")[-2]
    if not env:
        logger.warning(
            f"Could not load env.json for op.setup_wethyd in {cwd}. Skipping."
        )
        return Result.WARN

    if Path(tpr).exists() and not force:
        logger.info(f"Found existing {tpr} in {cwd}. Skipping.")
        return Result.SKIP

    gro = f"{job}-start.gro"
    top = f"{job}.top"

    dep = "eq"
    parent_gro = f"{dep}.gro"
    parent_top = f"{dep}.top"
    create_wethyd_top_and_gro(
        parent_top=Path(parent_top),
        parent_gro=Path(parent_gro),
        out_top=Path(top),
        out_gro=Path(gro),
        force=force,
    )

    write_qm_index(job, force)

    tpr = f"{job}.tpr"
    if Path(tpr).exists() and not force:
        logger.info(f"Found existing {tpr} in {cwd}. Skipping.")
        return Result.WARN

    charge = str(int(env[f"charge"]) - 1)

    generate_qm_reference(charge=charge, job=job, force=force)

    basis_set = env.get("basis_set", "DZVP-MOLOPT-GTH")
    xc_functional = env.get("xc_functional", "PBE")
    cp2k_inp = f"{job}.inp"

    cp2k_reference_to_inp(
        in_reference=f"{job}-qm-reference_cp2k.inp",
        in_template=None,
        out_cp2k_inp=cp2k_inp,
        job=job,
        basis_set=basis_set,
        xc_functional=xc_functional,
    )
    external_force = env[f"{system}_external_force"]
    rotation_restraints = get_rotation_restraints(ASSETS, system)
    fill_template(
        Path(f"{ASSETS}/{job}.template.mdp"),
        Path(f"{job}.mdp"),
        CHARGE=charge,
        EXTERNAL_FORCE=external_force,
        ROTATION_RESTRAINTS=rotation_restraints,
        PREFIX=job,
    )

    sh(
        f"gmx grompp -n {job}.ndx -f {job}.mdp -c {gro} -r {gro} -qmi {cp2k_inp} -p {job}.top -o {tpr} -maxwarn 2",
        quiet=1,
    )

    return Result.OK


def setup_wethyd_recovery(force):
    parent_job = "wethyd"
    job = f"{parent_job}-recovery"
    env = read_env()
    cwd = os.getcwd()
    system = cwd.split("/")[-2]
    if not env:
        logger.warning(
            f"Could not load env.json for op.setup_wethyd in {cwd}. Skipping."
        )
        return

    gmx_log = ut.shget(f"tail -n 50 {parent_job}.log")
    cp2k_log = ut.shget(f"tail -n 50 {parent_job}_cp2k.out")
    if Path(f"{parent_job}-slurm.out").exists():
        slurm_out = ut.shget(f"tail -n 50 {parent_job}-slurm.out")
    else:
        slurm_out = ""
    if "ABORT" in cp2k_log:
        logger.info(f"Found ABORT in {parent_job}_cp2k.out")
        logger.info(ut.shget(f"grep -B 4 -A 6 'ABORT' {parent_job}_cp2k.out"))
        logger.info(f"Attempting recovery")
    if "Fatal error" in slurm_out:
        logger.info(f"Found Fatal error in {parent_job}-slurm.out.")
        logger.info(ut.shget(f"grep -B 2 -A 6 'Fatal error' {parent_job}-slurm.out"))
        logger.info(f"Attempting recovery")
    if "Fatal error:" in gmx_log:
        logger.info(f"Found Fatal error in {parent_job}.log")
        logger.info(ut.shget(f"grep -B 2 -A 2 'Fatal error' {parent_job}.log"))
        logger.info(f"Simulation finished, probably by reaching its goal.")
        logger.info(f"No recovery needed. Skipping")
        return

    tpr = f"{job}.tpr"
    if Path(tpr).exists() and not force:
        logger.info(f"Found existing {tpr} in {cwd}. Skipping.")
        return

    gro = f"{job}-start.gro"

    # dump out a new gro file to start from
    frame, time = ut.find_latest_t(parent_job)
    if frame is None or time is None:
        logger.info(f"Did not find latest time in {parent_job}.log in {cwd}. Skipping.")
        return

    dt = 0.0002
    steps_from_end = 100
    t = time - dt * steps_from_end
    if t < 0:
        t = 0

    # taking from xtc (not trr) on purpose to introduce slight numerical fluctuation
    # from the rounding in hopes of recovering a nicer point in the energy landscape
    logger.info(f"Extracting frame at time {t}. From xtc.")
    sh(
        f"echo '0\n' | gmx trjconv -s {parent_job}.tpr -f {parent_job}.xtc -o {gro} -dump {t}"
    )

    # link shared assets between parent_job and recovery job
    ut.ln(f"{parent_job}.ndx", f"{job}.ndx")
    ut.ln(f"{parent_job}.top", f"{job}.top")
    env[f"{job}-recovery-start-t"] = t
    write_env(env)

    charge = str(int(env[f"charge"]) - 1)
    generate_qm_reference(charge=charge, job=job)

    cp2k_reference_to_inp(
        in_reference=f"{job}-qm-reference_cp2k.inp",
        in_template=None,
        out_cp2k_inp=f"{job}.inp",
        job=job,
    )

    external_force = env[f"{system}_external_force"]
    rotation_restraints = get_rotation_restraints(ASSETS, system)
    fill_template(
        Path(f"{ASSETS}/{parent_job}.template.mdp"),
        Path(f"{job}.mdp"),
        CHARGE=charge,
        EXTERNAL_FORCE=external_force,
        ROTATION_RESTRAINTS=rotation_restraints,
        PREFIX=job,
    )

    sh(
        f"gmx grompp -n {job}.ndx -f {job}.mdp -c {gro} -r {gro} -qmi {job}.inp -p {job}.top -o {tpr} -maxwarn 2"
    )


def write_qm_index(job: str, force: bool = False) -> Result:
    """ """
    env = read_env()
    if env is None:
        log_waiting("env.json", "write_qm_index")
        return Result.WAIT

    ndx = f"{job}.ndx"
    if Path(ndx).exists() and not force:
        logger.info(f"Found existing {ndx}. Skipping.")
        return Result.SKIP
    gro = f"{job}-start.gro"
    make_ndx(gro, ndx)
    add_terminal_index_groups(gro, ndx)

    if not Path(ndx).exists():
        raise FileNotFoundError(ndx)

    if job.startswith("wethyd") or job.startswith("wetbreak"):
        ixs_qm = env["ixs_qm_wethyd"]
    else:
        ixs_qm = env["ixs_qm"]

    ids_qmatoms = [str(int(x) + 1) for x in ixs_qm]
    id_c_carbonyl = str(int(env["ix_c_carbonyl"]) + 1)
    id_n_peptide = str(int(env["ix_n_peptide"]) + 1)

    with open(ndx, "a") as f:
        f.write("[ QMAtoms ]\n")
        f.writelines("\n".join(ids_qmatoms))
        f.write("\n")
        f.write("[ C_CARBONYL ]\n")
        f.write(id_c_carbonyl + "\n")
        f.write("[ N_PEPTIDE ]\n")
        f.write(id_n_peptide + "\n")
        f.write("[ PEPTIDE_BOND ]\n")
        f.write(id_c_carbonyl + " " + id_n_peptide + "\n")

        if "hyd" in job or "break" in job:
            ix_o_oh = env.get("ix_o_oh")
            if not ix_o_oh:
                log_waiting("find water and make hyd gro and top", "write_qm_index")
                return Result.WAIT
            id_o_oh = str(int(ix_o_oh) + 1)
            f.write("[ O_OH ]\n")
            f.write(id_o_oh + "\n")
        if "wethyd" in job or "wetbreak" in job:
            ixs_qw = env.get("ixs_qw")
            if not ixs_qw:
                log_waiting("find water and make wethyd gro and top", "write_qm_index")
                return Result.WAIT
            ixs_oh = env.get("ixs_oh")
            if not ixs_oh:
                log_waiting("find water and make hyd gro and top", "write_qm_index")
                return Result.WAIT
            ids_oh = [str(int(ix) + 1) for ix in ixs_oh]
            ids_qw = [str(int(x) + 1) for x in ixs_qw]
            f.write("[ QW ]\n")
            f.writelines("\n".join(ids_qw))
            f.write("\n")
            f.write("[ QWOH ]\n")
            f.writelines("\n".join(ids_qw + ids_oh))
            f.write("\n")
            f.write("[ OH ]\n")
            f.writelines("\n".join(ids_oh))
        f.write("\n")

    return Result.OK


def write_analysis_ndx(force: bool = False) -> Result:
    """ """
    job = "wethyd"
    env = read_env()
    if env is None:
        log_waiting("env.json", "write_qm_index")
        return Result.WAIT

    ndx = f"analysis.ndx"
    if Path(ndx).exists() and not force:
        logger.info(f"Found existing {ndx}. Skipping.")
        return Result.SKIP
    gro = f"{job}-start.gro"
    make_ndx(gro, ndx)
    add_terminal_index_groups(gro, ndx)
    cmd = f'gmx select -s {job}.tpr -f {job}-start.gro -n {ndx} -on {ndx}.tmp -select \'"ProtH" group "Protein" and type "H*"\''
    sh(cmd, quiet=0)
    sh(f"cat {ndx}.tmp >> {ndx}", quiet=1)
    os.remove(f"{ndx}.tmp")

    if not Path(ndx).exists():
        raise FileNotFoundError(ndx)

    ixs_qm = env["ixs_qm_wethyd"]
    ids_qmatoms = [str(int(x) + 1) for x in ixs_qm]
    id_c_carbonyl = str(int(env["ix_c_carbonyl"]) + 1)
    id_n_peptide = str(int(env["ix_n_peptide"]) + 1)
    id_o_carbonyl = str(int(id_c_carbonyl) + 1)
    ix_o_oh = env["ix_o_oh"]
    id_o_oh = str(int(ix_o_oh) + 1)

    with open(ndx, "a") as f:
        f.write("[ QMAtoms ]\n")
        f.writelines("\n".join(ids_qmatoms))
        f.write("\n")
        f.write("[ C_CARBONYL ]\n")
        f.write(id_c_carbonyl + "\n")
        f.write("[ O_CARBONYL ]\n")
        f.write(id_o_carbonyl + "\n")
        f.write("[ N_PEPTIDE ]\n")
        f.write(id_n_peptide + "\n")
        f.write("[ PEPTIDE_BOND ]\n")
        f.write(id_c_carbonyl + " " + id_n_peptide + "\n")
        f.write("[ O_OH ]\n")
        f.write(id_o_oh + "\n")

        ixs_qw = env["ixs_qw"]
        ixs_oh = env["ixs_oh"]
        ids_oh = [str(int(ix) + 1) for ix in ixs_oh]
        ids_qw = [str(int(x) + 1) for x in ixs_qw]
        f.write("[ QW ]\n")
        f.writelines("\n".join(ids_qw))
        f.write("\n")
        f.write("[ QWOH ]\n")
        f.writelines("\n".join(ids_qw + ids_oh))
        f.write("\n")
        f.write("[ OH ]\n")
        f.writelines("\n".join(ids_oh))
        f.write("\n")

    return Result.OK


def generate_qm_reference(
    charge: str, job: str, parent_job: str | None = None, force: bool = False
) -> Result:
    """ """
    logger.info(f"Generating qm reference for {job}")
    env = read_env()
    if env is None:
        raise ValueError("Something went terribly wrong")
    mdp = f"{job}-qm-reference.mdp"
    ndx = f"{job}.ndx"
    gro = f"{job}-start.gro"
    top = f"{job}.top"
    reference_job = f"{job}-qm-reference"

    if parent_job is not None:
        ndx = f"{parent_job}.ndx"
        top = f"{parent_job}.top"

    xc_functional = env.get("xc_functional", "PBE")
    if xc_functional == "B3LYP":
        method = "BLYP"
    elif xc_functional == "PBE0":
        method = "PBE"
    else:
        method = "PBE"

    fill_template(
        Path(f"{ASSETS}/qm-reference.template.mdp"),
        Path(mdp),
        CHARGE=charge,
        PREFIX=reference_job,
        METHOD=method,
    )
    reference_inp = Path(f"{reference_job}_cp2k.inp")
    if reference_inp.exists() and not force:
        log_skipped(str(reference_inp), f"generating cp2k reference inp")
        return Result.SKIP
    if reference_inp.exists() and force:
        os.remove(reference_inp)
    sh(
        f"gmx grompp -n {ndx} -f {mdp} -c {gro} -r {gro} -p {top} -o {reference_job}.tpr -maxwarn 2",
        quiet=0,
    )
    process = sp.Popen(
        f"./local-job.sh {reference_job}",
        stdout=sp.PIPE,
        shell=True,
        preexec_fn=os.setsid,
    )
    pid = process.pid
    while not reference_inp.exists():
        time.sleep(1)

    for _ in range(3):
        time.sleep(1)
        os.killpg(os.getpgid(pid), signal.SIGKILL)

    logger.info(f"Generated qm reference: {reference_job}_cp2k.inp")
    return Result.OK


def center_xtc(job: str, force: bool = False) -> Result:
    """ """
    env = read_env()
    assert env
    cwd = os.getcwd()
    if "-conf-" in job:
        # symlink index file from regular job
        parent_job = job.split("-conf-")[0]
        file = f"{parent_job}.ndx"
        target = f"{job}.ndx"
        sh(f"ln -srf {file} {target}")

    xtc = f"{job}.xtc"
    gro_out = f"{job}-center.gro"
    xtc_out = f"{job}-center.xtc"
    if not Path(xtc).exists():
        log_waiting(xtc, "center xtc")
        return Result.WAIT
    if not Path(f"{job}.tpr").exists():
        log_waiting(f"{job}.tpr", "center xtc")
        return Result.WAIT
    if not Path(f"{job}.ndx").exists():
        log_waiting(f"{job}.ndx", "center xtc")
        return Result.WAIT
    if Path(gro_out).exists() and Path(xtc_out).exists() and not force:
        log_skipped(f"{gro_out} and {xtc_out}", "center xtc")
        return Result.SKIP

    logger.info(f"centering {job} in {cwd}")

    write_qm_index(job)
    logger.info(f"Writing first centered frame as gro")
    sh(
        f"echo 'Protein\n non-Water' | gmx trjconv -f {xtc} -s {job}.tpr -pbc mol -dump -1 -n {job}.ndx -center -o {gro_out}",
        quiet=2,
    )
    logger.info(f"Centering xtc to Protein")
    sh(
        f"echo 'Protein\n non-Water' | gmx trjconv -f {xtc} -s {job}.tpr -pbc mol -n {job}.ndx -center -o {xtc_out}",
        quiet=2,
    )
    return Result.OK


def analyse_ti_proton_distances(force: bool = False) -> None:
    env = read_env()
    cwd = os.getcwd()
    parentjob = "wethyd"
    assert env, f"no env.json in {cwd}"
    times = env.get(f"{parentjob}-config-times")
    assert times, f"no {parentjob}-config-times in {cwd}"
    ixs_qw = env.get("ixs_qw")
    ixs_oh = env.get("ixs_oh")
    ix_o_carbonyl = env.get("ix_o_carbonyl")
    ix_n_peptide = env.get("ix_n_peptide")
    n_qm_water = env.get("n_qm_water")
    assert ixs_qw, f"no ixs_qw in {cwd}"
    assert ixs_oh, f"no ixs_oh in {cwd}"
    assert ix_o_carbonyl, f"no ix_o_carbonyl in {cwd}"
    assert ix_n_peptide, f"no ix_n_peptide in {cwd}"
    assert n_qm_water, f"no n_qm_water in {cwd}"
    # order is O,H1,H2 until 3*n_qm_water
    ids_o = []
    ids_h = []
    for i, ix in enumerate(ixs_qw):
        ix = int(ix)
        id = ix + 1
        if i % 3 == 0:
            ids_o.append(id)
        else:
            ids_h.append(id)
    ids_o.append(int(ixs_oh[0]) + 1)  # add O from original OH
    ids_o.append(int(ix_o_carbonyl) + 1) # add carbonyl O
    ids_o.append(int(ix_n_peptide) + 1) # add peptide N to the proton acceptors

    ids_h.append(int(ixs_oh[1]) + 1)  # add H from original OH

    s = f"atomnr {' '.join(map(str, ids_o))};"
    with open("protdist-o.sel", "w") as f:
        f.write(s)

    s = f"atomnr {' '.join(map(str, ids_h))};"
    with open("protdist-h.sel", "w") as f:
        f.write(s)

    for time in times:
        job = f"{parentjob}-conf-{time}"
        target = f"{job}-protdist.xvg"
        if not Path(f"{job}.xtc").exists():
            log_waiting(f"{job}.xtc", f"in {cwd}")
            continue
        if Path(target).exists() and not force:
            logger.info(f"Skipping {target} as it already exists.")
            continue
        cmd = f"gmx pairdist -f {job}.xtc -s {job}.tpr -ref -sf protdist-o.sel -sel -sf protdist-h.sel -refgrouping none -selgrouping none -o {target}"
        logger.info(f"Running pairdist")
        try:
            sh(cmd, quiet=0)
        except sp.CalledProcessError as e:
            logger.error(f"Error running pairdist: {e}")

def analyse_ti_protonation(force: bool = False) -> None:
    """
    uses <https://manual.gromacs.org/documentation/current/onlinehelp/gmx-pairdist.html>
    """
    cwd = os.getcwd()
    env = read_env()

    refpath = f"tidist-o.sel"
    selpath = f"tidist-h.sel"
    parentjob = "wethyd"

    assert env, f"no env.json in {cwd}"
    times = env.get(f"{parentjob}-config-times")
    assert times, f"no {parentjob}-config-times in {cwd}"
    ixs_qw = env.get("ixs_qw")
    ixs_oh = env.get("ixs_oh")
    ix_c_carbonyl = env.get("ix_c_carbonyl")
    ix_o_carbonyl = env.get("ix_o_carbonyl")
    n_qm_water = env.get("n_qm_water")
    assert ixs_qw, f"no ixs_qw in {cwd}"
    assert ixs_oh, f"no ixs_oh in {cwd}"
    assert ix_c_carbonyl, f"no ix_c_carbonyl in {cwd}"
    assert ix_o_carbonyl, f"no ix_o_carbonyl in {cwd}"
    assert n_qm_water, f"no n_qm_water in {cwd}"

    # order is O,H1,H2 until 3*n_qm_water
    ids_h = []
    for i, ix in enumerate(ixs_qw):
        ix = int(ix)
        id = ix + 1
        if i % 3 == 0:
            pass
        else:
            ids_h.append(id)
    ids_h.append(int(ixs_oh[1]) + 1)  # add H from original OH
    ix_o_carbonyl = int(ix_o_carbonyl)
    id_o_carbonyl = ix_o_carbonyl + 1

    s = f"atomnr {id_o_carbonyl};"
    with open(refpath, "w") as f:
        f.write(s)
    s = f"atomnr {' '.join(map(str, ids_h))};\n"
    s += f'group "Protein" and type H;\n'
    s += f"resname SOL and type HW;"
    with open(selpath, "w") as f:
        f.write(s)

    for time in times:
        job = f"{parentjob}-conf-{time}"
        target = f"{job}-tidist.xvg"
        if not Path(f"{job}.xtc").exists():
            log_waiting(f"{job}.xtc", f"in {cwd}")
            continue
        if Path(target).exists() and not force:
            logger.info(f"Skipping {target} as it already exists.")
            continue
        cmd = f"gmx pairdist -f {job}.xtc -s {job}.tpr -ref -sf {refpath} -sel -sf {selpath} -refgrouping none -selgrouping all -o {target} -cutoff 0.5"
        logger.info(f"Running pairdist")
        try:
            sh(cmd)
        except sp.CalledProcessError as e:
            logger.error(f"Error running pairdist: {e}")

def analyse_ti_stability(force: bool = False) -> None:
    """
    uses <https://manual.gromacs.org/documentation/current/onlinehelp/gmx-pairdist.html>
    """
    cwd = os.getcwd()
    env = read_env()

    refpath = f"ti-stability-c.sel"
    selpath = f"ti-stability-o.sel"
    parentjob = "wethyd"

    assert env, f"no env.json in {cwd}"
    times = env.get(f"{parentjob}-config-times")
    assert times, f"no {parentjob}-config-times in {cwd}"
    ixs_oh = env.get("ixs_oh")
    ix_c_carbonyl = env.get("ix_c_carbonyl")
    ix_o_carbonyl = env.get("ix_o_carbonyl")
    assert ixs_oh, f"no ixs_oh in {cwd}"
    assert ix_c_carbonyl, f"no ix_c_carbonyl in {cwd}"
    assert ix_o_carbonyl, f"no ix_o_carbonyl in {cwd}"

    id_c_carbonyl = int(ix_c_carbonyl) + 1
    id_o_carbonyl = int(ix_o_carbonyl) + 1
    id_o_oh = int(ixs_oh[0]) + 1

    s = f"atomnr {id_c_carbonyl};"
    with open(refpath, "w") as f:
        f.write(s)

    s = f"atomnr {id_o_carbonyl} {id_o_oh};\n"
    with open(selpath, "w") as f:
        f.write(s)

    for time in times:
        job = f"{parentjob}-conf-{time}"
        target = f"{job}-ti-stability.xvg"
        if not Path(f"{job}.xtc").exists():
            log_waiting(f"{job}.xtc", f"in {cwd}")
            continue
        if Path(target).exists() and not force:
            logger.info(f"Skipping {target} as it already exists.")
            continue
        cmd = f"gmx pairdist -f {job}.xtc -s {job}.tpr -ref -sf {refpath} -sel -sf {selpath} -refgrouping none -selgrouping none -o {target} -cutoff 0.5"
        logger.info(f"Running pairdist")
        try:
            sh(cmd)
        except sp.CalledProcessError as e:
            logger.error(f"Error running pairdist: {e}")


def analyse_break_ti_distances(force: bool = False) -> None:
    """
    uses <https://manual.gromacs.org/documentation/current/onlinehelp/gmx-pairdist.html>
    """
    cwd = os.getcwd()
    env = read_env()

    refpath = f"break-ti-distances-ref.sel"
    selpath = f"break-ti-distances-sel.sel"
    parentjob = "wetbreak"

    assert env, f"no env.json in {cwd}"
    times = env.get(f"{parentjob}-config-times")
    if times is None:
        logger.warning(f"no {parentjob}-config-times in {cwd}. Skipping.")
        return
    ixs_oh = env.get("ixs_oh")
    ix_c_carbonyl = env.get("ix_c_carbonyl")
    ix_o_carbonyl = env.get("ix_o_carbonyl")
    ix_n_peptide = env.get("ix_n_peptide")
    assert ix_n_peptide, f"no ix_n_peptide in {cwd}"
    assert ixs_oh, f"no ixs_oh in {cwd}"
    assert ix_c_carbonyl, f"no ix_c_carbonyl in {cwd}"
    assert ix_o_carbonyl, f"no ix_o_carbonyl in {cwd}"

    id_c_carbonyl = int(ix_c_carbonyl) + 1
    id_o_carbonyl = int(ix_o_carbonyl) + 1
    id_o_oh = int(ixs_oh[0]) + 1
    id_n_peptide = int(ix_n_peptide) + 1

    s = f"atomnr {id_c_carbonyl};"
    with open(refpath, "w") as f:
        f.write(s)

    s = f"atomnr {id_o_carbonyl} {id_o_oh} {id_n_peptide};\n"
    with open(selpath, "w") as f:
        f.write(s)

    for time in times:
        job = f"{parentjob}-conf-{time}"
        target = f"{job}-break-ti-distances.xvg"
        if not Path(f"{job}.xtc").exists():
            log_waiting(f"{job}.xtc", f"in {cwd}")
            continue
        if Path(target).exists() and not force:
            logger.info(f"Skipping {target} as it already exists.")
            continue
        cmd = f"gmx pairdist -f {job}.xtc -s {job}.tpr -ref -sf {refpath} -sel -sf {selpath} -refgrouping none -selgrouping none -o {target} -cutoff 0.5"
        logger.info(f"Running pairdist")
        try:
            sh(cmd)
        except sp.CalledProcessError as e:
            logger.error(f"Error running pairdist: {e}")

def analyse_break_proton_distances(force: bool = False) -> None:
    """
    Record all distances between protons (from QM water, OH and the protein)
    and oxygens (from QM water, OH and the carbonyl oxygen) and the peptide bond Nitrogen
    """
    env = read_env()
    cwd = os.getcwd()
    parentjob = "wetbreak"
    refpath = "protdist-break-o.sel"
    selpath = "protdist-break-h.sel"

    assert env, f"no env.json in {cwd}"
    times = env.get(f"{parentjob}-config-times")
    if times is None:
        # logger.warning(f"no {parentjob}-config-times in {cwd}. Skipping.")
        return
    ixs_qw = env.get("ixs_qw")
    ixs_oh = env.get("ixs_oh")
    ix_o_carbonyl = env.get("ix_o_carbonyl")
    ix_n_peptide = env.get("ix_n_peptide")
    n_qm_water = env.get("n_qm_water")
    assert ixs_qw, f"no ixs_qw in {cwd}"
    assert ixs_oh, f"no ixs_oh in {cwd}"
    assert ix_o_carbonyl, f"no ix_o_carbonyl in {cwd}"
    assert ix_n_peptide, f"no ix_n_peptide in {cwd}"
    assert n_qm_water, f"no n_qm_water in {cwd}"
    # order is O,H1,H2 until 3*n_qm_water
    ids_o = []
    ids_h = []
    for i, ix in enumerate(ixs_qw):
        ix = int(ix)
        id = ix + 1
        if i % 3 == 0:
            ids_o.append(id)
        else:
            ids_h.append(id)
    ids_o.append(int(ixs_oh[0]) + 1)  # add O from original OH
    ids_o.append(int(ix_o_carbonyl) + 1) # add carbonyl O
    ids_o.append(int(ix_n_peptide) + 1) # add peptide N to the proton acceptors

    ids_h.append(int(ixs_oh[1]) + 1)  # add H from original OH
    # add the H from the N-H of the peptide bond
    ids_h.append(int(ix_n_peptide) + 1)

    s = f"atomnr {' '.join(map(str, ids_o))};"
    with open(refpath, "w") as f:
        f.write(s)

    s = f"atomnr {' '.join(map(str, ids_h))};"
    with open(selpath, "w") as f:
        f.write(s)

    for time in times:
        job = f"{parentjob}-conf-{time}"
        target = f"{job}-protdist.xvg"
        if not Path(f"{job}.xtc").exists():
            log_waiting(f"{job}.xtc", f"in {cwd}")
            continue
        if Path(target).exists() and not force:
            logger.info(f"Skipping {target} as it already exists.")
            continue
        cmd = f"gmx pairdist -f {job}.xtc -s {job}.tpr -ref -sf {refpath} -sel -sf {selpath} -refgrouping none -selgrouping none -o {target}"
        logger.info(f"Running pairdist")
        try:
            sh(cmd, quiet=0)
        except sp.CalledProcessError as e:
            logger.error(f"Error running pairdist: {e}")

def analyse_distances(job: str, force: bool = False) -> Result:
    env = read_env()
    assert env
    target = f"{job}-mindist-o-h.xvg"
    target2 = f"{job}-mindist-o-c.xvg"
    target3 = f"{job}-mindist-c-w.xvg"
    if (
        Path(target).exists()
        and Path(target2).exists()
        and Path(target3).exists()
        and not force
    ):
        log_skipped(target, f"analyse_distances of {job}")
        return Result.SKIP

    write_qm_index(job)
    link(f"{ASSETS}/analyse-distances.sh")
    sh(f"./analyse-distances.sh {job}")
    return Result.OK


def get_energy(job: str, force: bool = False) -> Result:
    target = f"{job}-energy.xvg"
    if Path(target).exists() and not force:
        log_skipped(target, f"skipped get_energy for {job}")
        return Result.SKIP
    try:
        sh(
            f"echo '1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 41 0' | gmx energy -f {job}.edr -o {target}"
        )
        logger.info(f"Wrote energy for {job}.")
    except sp.CalledProcessError as e:
        logger.error(f"Error getting energy for {job}: {e}")
        pass
    return Result.OK


def vis_job(
    job: str,
    centered: bool = True,
    open: bool = False,
    single_frame: bool = False,
    glxfix=False,
    force: bool = False,
) -> Result:
    env = read_env()
    assert env
    if not env:
        log_waiting("env.json", f"{job}")
        return Result.WAIT
    qm_indices = " ".join(env[f"ixs_qm"]) + " or resname OH or resname W"
    vmdfile = f"{job}.vmd"
    if Path(vmdfile).exists() and not force and not open:
        log_skipped(vmdfile, f"{job}")
        return Result.SKIP
    if centered:
        center = "-center"
    else:
        center = ""

    if glxfix:
        fix = "__GLX_VENDOR_LIBRARY_NAME=mesa "
    else:
        fix = ""

    gro = f"{job}{center}.gro"
    xtc = f"{job}{center}.xtc"
    if not Path(gro).exists() or not Path(xtc).exists():
        log_waiting(gro, f"{job}")
        return Result.WAIT
    parent_job = job
    if "-conf-" in job:
        parent_job = job.split("-conf-")[0]
    if "wet" in job:
        # remove `Q` from `{job}-center.gro`
        # as it is not recognized by vmd
        # QW turns into W
        sh(f"sed -i 's/Q//g' {gro}")

    if single_frame:
        xtc = ""
        gro = f"{job}.gro"
    if not Path(xtc).exists():
        xtc = ""
    if not Path(gro).exists():
        log_waiting(gro, f"{job}")
        return Result.WAIT
    fill_template(
        Path(f"{ASSETS}/vis-{parent_job}.template.vmd"),
        Path(vmdfile),
        GRO=gro,
        XTC=xtc,
        QM_INDICES=qm_indices,
    )
    if open:
        sh(f"{fix} vmd -e {vmdfile}", quiet=0)

    return Result.OK

def vis_gro(
    job: str,
    gro: str,
    center: bool = True,
    open: bool = True,
    glxfix=True,
    force: bool = False,
) -> Result:
    env = read_env()
    assert env
    vmdfile = f"{gro}.vmd"

    if Path(vmdfile).exists() and not force and not open:
        log_skipped(vmdfile, f"{gro}")
        return Result.SKIP

    if glxfix:
        fix = "__GLX_VENDOR_LIBRARY_NAME=mesa "
    else:
        fix = ""

    if not Path(gro).exists():
        log_waiting(gro, f"{job}")
        return Result.WAIT

    center_gro = f"{gro}-center.gro"
    new_gro = f"{gro}-tmp.gro"
    if center:
        if not Path(center_gro).exists() or force:
            sh(
                f"echo 'Protein\n non-Water' | gmx trjconv -f {gro} -s {job}.tpr -pbc mol -n {job}.ndx -center -o {center_gro}",
                quiet=0,
            )
            sh(f"sed 's/Q//g' {center_gro} > {new_gro}")
    else:
        sh(f"sed 's/Q//g' {gro} > {new_gro}")

    qm_selector = "index " + " ".join(env[f"ixs_qm"]) + " or resname OH or resname W"
    fill_template(
        Path(f"{ASSETS}/vis-{job}-gro.template.vmd"),
        Path(vmdfile),
        GRO=new_gro,
        QM_SELECTOR=qm_selector,
    )
    if open:
        sh(f"{fix} vmd -e {vmdfile}", quiet=0)

    return Result.OK

def vis_eq_gro(
    gro: str,
    selector: str,
    center: bool = True,
    open: bool = True,
    glxfix=True,
    force: bool = False,
) -> Result:
    vmdfile = f"{gro}.vmd"
    if glxfix:
        fix = "__GLX_VENDOR_LIBRARY_NAME=mesa "
    else:
        fix = ""

    if not Path(gro).exists():
        log_waiting(gro, gro)
        return Result.WAIT

    center_gro = f"{gro}-center.gro"
    # top = gro.replace(".gro", ".top")

    if center and not Path(center_gro).exists() and not force:
        sh(
            f"echo 'Protein\n non-Water' | gmx trjconv -f {gro} -s {gro} -pbc mol -center -o {center_gro}",
            quiet=0,
        )
    if not center:
        center_gro = gro

    fill_template(
        Path(f"{ASSETS}/vis-wethyd-gro.template.vmd"),
        Path(vmdfile),
        GRO=center_gro,
        QM_SELECTOR=selector,
    )
    if open:
        sh(f"{fix} vmd -e {vmdfile}", quiet=0)

    return Result.OK




def analyse_job(
    job: str, analysis_types: list[str] = ["all"], force: bool = False
) -> Result:
    if "distances" in analysis_types or "all" in analysis_types:
        safely(analyse_distances, job=job, force=force)
    if "energy" in analysis_types or "all" in analysis_types:
        safely(get_energy, job=job, force=force)
    if "center" in analysis_types or "all" in analysis_types:
        safely(center_xtc, job=job, force=force)
        safely(vis_job, job=job, force=force)

    return Result.OK


def center_vis(job: str, force=False, **kwargs) -> Result:
    safely(center_xtc, job=job, force=force)
    safely(vis_job, job=job, force=force, **kwargs)
    return Result.OK


def get_umbrella_window_times(
    parent_job: str,
    cutoff: float = 0.25,
    further: int = 20,
    closer: int = 30,
    force: bool = False,
    extend: bool = False,
) -> Result:
    env = read_env()
    cwd = os.getcwd()
    if not env:
        log_waiting("env.json", f"get_umbrella_window_times in {cwd}")
        return Result.WAIT

    og_times = env.get(f"{parent_job}-config-times", None)
    og_distances = env.get(f"{parent_job}-config-ds", None)
    if og_times is not None and len(og_times) >= (closer + further):
        log_skipped(
            f"{parent_job}-config-times with {len(og_times)} windows",
            f"get_umbrella_window_times for {parent_job} in {cwd}",
        )
        return Result.SKIP

    logger.info(f"Getting umbrella window times for {parent_job} in {cwd}")
    colnames = ["t", "x1", "x2", "d"]
    if parent_job == "wethyd" or parent_job == "wetbreak":
        colnames += ["dqw"]
    xs = read_xvg(f"{parent_job}_pullx.xvg", colnames)

    ds = []
    ts_start = xs.loc[xs["d"] > cutoff]
    if len(ts_start) > 0:
        ts_start = ts_start.assign(bin=pd.cut(ts_start["d"], bins=further))
        ts_start = ts_start.groupby("bin", observed=True)["t"].median().reset_index()
        ds += [round(b.mid, 4) for b in ts_start.bin.tolist()]
        ts_start = ts_start["t"].round(4).tolist()
    else:
        ts_start = []

    ts_end = xs.loc[xs["d"] <= cutoff]
    if len(ts_end) > 0:
        ts_end = ts_end.assign(bin=pd.cut(ts_end["d"], bins=closer))
        ts_end = ts_end.groupby("bin", observed=True)["t"].median().reset_index()
        ds += [round(b.mid, 4) for b in ts_end.bin.tolist()]
        ts_end = ts_end["t"].round(4).tolist()
    else:
        ts_end = []

    ts = ts_start + ts_end
    if extend and og_times and og_distances:
        logger.info(f"Extending us windows for {parent_job}")
        if parent_job == "wetbreak":
            # discard last times due to too large distance
            og_times = og_times[:-3]
            og_distances = og_distances[:-3]
        if parent_job == "wethyd":
            min_og_d = min([float(x) for x in og_distances])
            comparison = lambda x: x < min_og_d
        elif parent_job == "wetbreak":
            max_og_d = max([float(x) for x in og_distances])
            comparison = lambda x: x > max_og_d
        else:
            raise NotImplementedError(
                f"Can't find umbrella windows for parent_job {parent_job}"
            )
        ts_ds = [(t, d) for t, d in zip(ts, ds) if comparison(float(d))]
        if parent_job == "wetbreak":
            # discard last times due to too large distance
            ts_ds = ts_ds[:-3]

        new_times = og_times + [f"{t:0<6}" for t, _ in ts_ds]
        logger.info(f"Adding {len(ts_ds)} new times.")
        env[f"{parent_job}-config-times"] = new_times
        new_distances = og_distances + [f"{d:0<6}" for _, d in ts_ds]
        env[f"{parent_job}-config-ds"] = new_distances
    else:
        logger.info(f"Writing new us windows for {parent_job}")
        env[f"{parent_job}-config-times"] = [f"{t:0<6}" for t in ts]
        env[f"{parent_job}-config-ds"] = [f"{d:0<6}" for d in ds]
    write_env(env)

    return Result.OK


def extract_umbrella_configs(parent_job: str = "wethyd", force: bool = False) -> Result:
    env = read_env()
    cwd = os.getcwd()
    if not env:
        log_waiting("env.json", f"get_umbrella_window_times for {parent_job} in {cwd}")
        return Result.WAIT
    ts = env.get(f"{parent_job}-config-times")
    if not ts:
        log_waiting(
            f"{parent_job}-config-times",
            f"extract_umbrella_configs for {parent_job} in {cwd}",
        )
        return Result.WAIT
    for t in tqdm(ts):
        starting_gro = f"{parent_job}-conf-{t}-start.gro"
        if Path(starting_gro).exists() and not force:
            log_skipped(
                starting_gro, f"extract_umbrella_configs for {parent_job} in {cwd}"
            )
            continue
        sh(
            f"echo '0\n' | gmx trjconv -s {parent_job}.tpr -f {parent_job}.trr -o {starting_gro} -dump {t}"
        )

    return Result.OK


def setup_umbrella_sampling(parent_job: str = "wethyd", force: bool = False) -> Result:
    env = read_env()
    cwd = os.getcwd()
    if not env or env.get(f"{parent_job}-config-times") is None:
        log_waiting(
            "config times in env.json",
            f"setup_umbrella_sampling for {parent_job} in {cwd}",
        )
        return Result.WAIT
    ndx = f"{parent_job}.ndx"
    top = f"{parent_job}.top"
    mdp = f"{parent_job}-us.mdp"
    template = f"{ASSETS}/{parent_job}-us.template.mdp"
    ts = env[f"{parent_job}-config-times"]

    for t in tqdm(ts):
        job = f"{parent_job}-conf-{t}"

        # this can be commented out to regenerate tprs
        target = f"{job}.tpr"
        if Path(target).exists() and not force:
            log_skipped(target, f"setup_umbrella_sampling in {cwd}")
            continue

        charge = str(int(env[f"charge"]) - 1)

        generate_qm_reference(
            charge=charge, job=job, parent_job=parent_job, force=force
        )

        basis_set = env.get("basis_set", "DZVP-MOLOPT-GTH")
        xc_functional = env.get("xc_functional", "PBE")
        cp2k_inp = f"{job}.inp"

        cp2k_reference_to_inp(
            in_reference=f"{job}-qm-reference_cp2k.inp",
            in_template=None,
            out_cp2k_inp=cp2k_inp,
            job=job,
            basis_set=basis_set,
            xc_functional=xc_functional,
        )

        system = cwd.split("/")[-2]
        rotation_restraints = get_rotation_restraints(ASSETS, system)
        fill_template(
            Path(template),
            Path(mdp),
            EXTERNAL_FORCE=env[f"{system}_external_force"],
            ROTATION_RESTRAINTS=rotation_restraints,
            CHARGE=env["charge"],
            PREFIX=job,
        )

        # the 2 warnings (that are fine):
        # - charged QM system (already balanced by the MM)
        # - absolute pulling reference
        try:
            sh(
                f"gmx grompp -n {ndx} -f {mdp} -c {job}-start.gro -r {job}-start.gro -qmi {job}.inp -p {top} -o {job}.tpr -maxwarn 2",
                quiet=0
            )
        except sp.CalledProcessError as e:
            logger.error(f"Error in grompp for {job} in {cwd}: {e}")
            return Result.ERROR

    return Result.OK


def file_is_empty(path: Path):
    if path.stat().st_size == 0:
        return True
    else:
        return False


def analyse_wethyd_us(force: bool = False, to_discard: pd.DataFrame|None = None) -> Result:
    """ """
    parent_job = "wethyd"
    env = read_env()
    cwd = os.getcwd()
    if Path(f"{parent_job}-bsResult.xvg").exists() and not force:
        log_skipped(
            f"{parent_job}-bsResult.xvg", f"analyse_wethyd_us for {parent_job} in {cwd}"
        )
        return Result.SKIP
    if not env or env.get(f"{parent_job}-config-times") is None:
        log_waiting("env.json", f"analyse_umbrella_sampling for {parent_job} in {cwd}")
        return Result.WAIT

    ix_c = env['ix_c_carbonyl']
    ix_n = env['ix_n_peptide']
    cwd_split = cwd.split("/")
    system = cwd_split[-2]
    frame = cwd_split[-1].removeprefix("frame-")

    ts = [
        t
        for t in env[f"{parent_job}-config-times"]
        if Path(f"{parent_job}-conf-{t}_pullf.xvg").exists()
        and not file_is_empty(Path(f"{parent_job}-conf-{t}_pullf.xvg"))
    ]
    if len(ts) == 0:
        log_waiting(
            f"{parent_job}-conf-*-pullf.xvg",
            f"analyse_umbrella_sampling for {parent_job} in {cwd}",
        )
        return Result.WAIT

    if to_discard is not None:
        ts_to_drop = to_discard.query(f"system == '{system}' and frame == '{frame}' and ix_c == '{ix_c}' and ix_n == '{ix_n}'")['conf_t'].to_list()
        logger.info(f"Discarding {len(ts_to_drop)} windows for system {system} frame {frame} ix_c {ix_c} ix_n {ix_n}")
        len_og = len(ts)
        ts = [t for t in ts if t not in ts_to_drop]
        len_new = len(ts)
        if len_og != len_new:
            logger.info(f"Dropping {len_og - len_new} ts for {system} frame {frame} ix_c {ix_c} ix_n {ix_n} because of to_discard")

    tprs = [f"{parent_job}-conf-{t}.tpr\n" for t in ts]
    pullfs = [f"{parent_job}-conf-{t}_pullf.xvg\n" for t in ts]
    pullxs = [f"{parent_job}-conf-{t}_pullx.xvg\n" for t in ts]
    coords = ["0 0 1 0\n" for _ in ts]
    write(tprs, f"{parent_job}-tpr-files.dat")
    write(pullfs, f"{parent_job}-pullf-files.dat")
    write(pullxs, f"{parent_job}-pullx-files.dat")
    write(coords, f"{parent_job}-select-pull-coord.dat")
    # process = sh(
    #     f"gmx wham -it {parent_job}-tpr-files.dat -if {parent_job}-pullf-files.dat -is {parent_job}-select-pull-coord.dat -o {parent_job}-profile.xvg -hist {parent_job}-hist.xvg -bsres {parent_job}-bsResult.xvg -bsprof {parent_job}-bsProfs.xvg -unit kJ -bins 200 -b 0.1 -nBootstrap 50 -temp 310 -zprof0 0.8",
    #     capture=True,
    #     check=False,
    # )
    process = sh(
        f"gmx wham -it {parent_job}-tpr-files.dat -if {parent_job}-pullf-files.dat -is {parent_job}-select-pull-coord.dat -o {parent_job}-profile.xvg -hist {parent_job}-hist.xvg -bsres {parent_job}-bsResult.xvg -bsprof {parent_job}-bsProfs.xvg -unit kJ -bins 40 -b 0.005 -nBootstrap 5 -temp 310 -max 0.5 -min 0.12",
        capture=True,
        check=False,
    )
    if process.returncode != 0:
        logger.error(f"Failed to run gmx wham: {process.stderr} {process.stdout}")
        return Result.ERROR

    return Result.OK

def analyse_combined_us(parent_job: str,
                        choices: pd.DataFrame,
                        to_discard: pd.DataFrame|None,
                        max: float = 0.5,
                        min: float = 0.12,
                        bins: int = 80,
                        force: bool=False) -> None:
    choices["system_dir"] = (
        choices.cwd.str.split("/").str[0:-1].str.join("/")
    )
    for dir in choices.system_dir.unique():
        cwds = choices[choices.system_dir == dir].cwd
        tprs = []
        pullfs = []
        pullxs = []
        coords = []

        if Path(f"{dir}/{parent_job}-profile.xvg").exists() and not force:
            return
        for cwd in cwds:
            env = read_env(cwd)
            assert env

            ix_c = env['ix_c_carbonyl']
            ix_n = env['ix_n_peptide']
            cwd_split = cwd.split("/")
            system = cwd_split[-2]
            frame = cwd_split[-1].removeprefix("frame-")
            ts = env.get(f"{parent_job}-config-times")
            if ts is None:
                logger.warning(f"No config times for {cwd}, skipping")
                continue

            ts = [
                t
                for t in ts
                if Path(f"{dir}/frame-{frame}/{parent_job}-conf-{t}_pullf.xvg").exists()
                and not file_is_empty(
                    Path(f"{dir}/frame-{frame}/{parent_job}-conf-{t}_pullf.xvg")
                )
            ]

            if to_discard is not None:
                ts_to_drop = to_discard.query(f"system == '{system}' and frame == '{frame}' and ix_c == '{ix_c}' and ix_n == '{ix_n}'")['conf_t'].to_list()
                len_og = len(ts)
                ts = [t for t in ts if t not in ts_to_drop]
                len_new = len(ts)
                if len_og != len_new:
                    logger.info(f"Dropping {len_og - len_new} ts for {system} frame {frame} ix_c {ix_c} ix_n {ix_n} because of to_discard")

            tprs += [f"frame-{frame}/{parent_job}-conf-{t}.tpr\n" for t in ts]
            pullfs += [f"frame-{frame}/{parent_job}-conf-{t}_pullf.xvg\n" for t in ts]
            pullxs += [f"frame-{frame}/{parent_job}-conf-{t}_pullx.xvg\n" for t in ts]
            coords += ["0 0 1 0\n" for _ in ts]

        if len(pullfs) == 0:
            log_waiting(f"{parent_job} us windows", f"in {dir}")
            continue
        with pushd(dir):
            write(tprs, f"{parent_job}-tpr-files.dat")
            write(pullfs, f"{parent_job}-pullf-files.dat")
            write(pullxs, f"{parent_job}-pullx-files.dat")
            write(coords, f"{parent_job}-select-pull-coord.dat")
            # sh(
            #     f"gmx wham -it {parent_job}-tpr-files.dat -if {parent_job}-pullf-files.dat -is {parent_job}-select-pull-coord.dat -o {parent_job}-profile.xvg -hist {parent_job}-hist.xvg -bsres {parent_job}-bsResult.xvg -bsprof {parent_job}-bsProfs.xvg -unit kJ -bins 200 -b 0.1 -nBootstrap 50 -temp 310 -zprof0 0.8",
            #     quiet=2,
            # )
            sh(
                f"gmx wham -it {parent_job}-tpr-files.dat -if {parent_job}-pullf-files.dat -is {parent_job}-select-pull-coord.dat -o {parent_job}-profile.xvg -hist {parent_job}-hist.xvg -bsres {parent_job}-bsResult.xvg -bsprof {parent_job}-bsProfs.xvg -unit kJ -bins {bins} -b 0.05 -nBootstrap 5 -temp 310 -max {max} -min {min}",
                quiet=0,
            )
            logger.info(f"Finished WHAM of {dir}")


def analyse_wethyd_us_energies(force: bool = False) -> Result:
    """ """
    parent_job = "wethyd"
    env = read_env()
    cwd = os.getcwd()
    if Path(f"{parent_job}-bsResult.xvg").exists() and not force:
        log_skipped(
            f"{parent_job}-bsResult.xvg", f"analyse_wethyd_us for {parent_job} in {cwd}"
        )
        return Result.SKIP
    if not env or env.get(f"{parent_job}-config-times") is None:
        log_waiting("env.json", f"analyse_umbrella_sampling for {parent_job} in {cwd}")
        return Result.WAIT
    ts = [
        t
        for t in env[f"{parent_job}-config-times"]
        if Path(f"{parent_job}-conf-{t}_pullf.xvg").exists()
        and not file_is_empty(Path(f"{parent_job}-conf-{t}_pullf.xvg"))
    ]
    if len(ts) == 0:
        log_waiting(
            f"{parent_job}-conf-*-pullf.xvg",
            f"analyse_umbrella_sampling for {parent_job} in {cwd}",
        )
        return Result.WAIT

    for t in ts:
        job = f"{parent_job}-conf-{t}"
        get_energy(job=job, force=force)

    return Result.OK


def analyse_wetbreak_us(force: bool = False) -> Result:
    """ """
    parent_job = "wetbreak"
    env = read_env()
    cwd = os.getcwd()
    if Path(f"{parent_job}-bsResult.xvg").exists() and not force:
        log_skipped(
            f"{parent_job}-bsResult.xvg",
            f"analyse_wetbreak_us for {parent_job} in {cwd}",
        )
        return Result.SKIP
    if not env or env.get(f"{parent_job}-config-times") is None:
        log_waiting("env.json", f"analyse_umbrella_sampling for {parent_job} in {cwd}")
        return Result.WAIT
    ts = [
        t
        for t in env[f"{parent_job}-config-times"]
        if Path(f"{parent_job}-conf-{t}_pullf.xvg").exists()
        and not file_is_empty(Path(f"{parent_job}-conf-{t}_pullf.xvg"))
    ]
    if len(ts) == 0:
        log_waiting(
            f"{parent_job}-conf-*-pullf.xvg",
            f"analyse_umbrella_sampling for {parent_job} in {cwd}",
        )
        return Result.WAIT

    tprs = [f"{parent_job}-conf-{t}.tpr\n" for t in ts]
    pullfs = [f"{parent_job}-conf-{t}_pullf.xvg\n" for t in ts]
    pullxs = [f"{parent_job}-conf-{t}_pullx.xvg\n" for t in ts]
    coords = ["0 0 1 0\n" for _ in ts]
    write(tprs, f"{parent_job}-tpr-files.dat")
    write(pullfs, f"{parent_job}-pullf-files.dat")
    write(pullxs, f"{parent_job}-pullx-files.dat")
    write(coords, f"{parent_job}-select-pull-coord.dat")

    # give either pullf or pullx, not both. -ix {job}-pullx-files.dat
    # sh(
    #     f" gmx wham -it {parent_job}-tpr-files.dat -if {parent_job}-pullf-files.dat -is {parent_job}-select-pull-coord.dat -o {parent_job}-profile.xvg -hist {parent_job}-hist.xvg -bsres {parent_job}-bsResult.xvg -bsprof {parent_job}-bsProfs.xvg -unit kJ -bins 100 -b 0.1 -nBootstrap 50"
    # )
    bins = 80
    max = 0.6
    min = 0.12
    sh(
        f"gmx wham -it {parent_job}-tpr-files.dat -if {parent_job}-pullf-files.dat -is {parent_job}-select-pull-coord.dat -o {parent_job}-profile.xvg -hist {parent_job}-hist.xvg -bsres {parent_job}-bsResult.xvg -bsprof {parent_job}-bsProfs.xvg -unit kJ -bins {bins} -b 0.05 -nBootstrap 5 -temp 310 -max {max} -min {min}",
        quiet=0,
    )

    return Result.OK


def setup_wetbreak(force: bool = False, hard_reset: bool = False) -> Result:
    """
    Depends on st.setup_wethyd and st.start_wethyd
    """
    parent_job = "wethyd"
    job = "wetbreak"
    cwd = os.getcwd()
    env = read_env()
    if env is None:
        log_waiting("env.json", f"wetbreak for {parent_job} in {cwd}")
        return Result.WAIT

    if hard_reset and force:
        # remove all job files
        sh(f"rm -f {job}*")

    tpr = f"{job}.tpr"
    if Path(tpr).exists() and not force:
        logger.info(f"Found existing {tpr} in {cwd}. Skipping.")
        return Result.SKIP


    sh(f"cp {parent_job}.top {job}.top")

    gro = f"{job}-start.gro"
    ts = env[f"{parent_job}-config-times"]
    ds = env[f"{parent_job}-config-ds"]
    ts_ds = sorted([(t, d) for t, d in zip(ts, ds)], key=lambda x: float(x[1]))
    choice = None
    previous = (None, None)
    for t, d in ts_ds:
        if float(d) >= TI_TARGET_DISTANCE:
            # use the time that is just below the target distance
            choice = previous
            break
        previous = (t, d)

    if choice is None:
        raise ValueError(f"Could not find a distance greater than {TI_TARGET_DISTANCE}")
    t, d = choice
    logger.info(f"Using configuration at time {t} with distance {d}.")
    sh(f"cp {parent_job}-conf-{t}-start.gro {gro}")
    env["wetbreak-start-t"] = t
    env["wetbreak-start-d"] = d
    write_env(env)

    charge = str(int(env[f"charge"]) - 1)
    write_qm_index(job, force)
    generate_qm_reference(charge=charge, job=job, force=force)

    basis_set = env.get("basis_set")
    if basis_set is None:
        raise ValueError("No basis set found in env.json")
    xc_functional = env.get("xc_functional", "PBE")
    cp2k_inp = f"{job}.inp"

    cp2k_reference_to_inp(
        in_reference=f"{job}-qm-reference_cp2k.inp",
        in_template=None,
        out_cp2k_inp=cp2k_inp,
        job=job,
        basis_set=basis_set,
        xc_functional=xc_functional,
    )

    system = cwd.split("/")[-2]
    external_force = env[f"{system}_external_force"]
    rotation_restraints = get_rotation_restraints(ASSETS, system)
    fill_template(
        Path(f"{ASSETS}/{job}.template.mdp"),
        Path(f"{job}.mdp"),
        CHARGE=charge,
        EXTERNAL_FORCE=external_force,
        ROTATION_RESTRAINTS=rotation_restraints,
        PREFIX=job,
    )
    sh(
        f"gmx grompp -n {job}.ndx -f {job}.mdp -c {gro} -r {gro} -qmi {cp2k_inp} -p {job}.top -o {tpr} -maxwarn 2"
    )
    logger.info(f"Generated {tpr} for {job} in {cwd}")
    return Result.OK

def setup_wetbreak_restrained(force: bool = False, hard_reset: bool = False) -> Result:
    """
    Depends on st.setup_wethyd and st.start_wethyd
    """
    parent_job = "wethyd"
    job = "wetbreak_restrained"
    cwd = os.getcwd()
    env = read_env()
    if env is None:
        log_waiting("env.json", f"wetbreak for {parent_job} in {cwd}")
        return Result.WAIT

    if hard_reset and force:
        # remove all job files
        sh(f"rm -f {job}*")

    tpr = f"{job}.tpr"
    if Path(tpr).exists() and not force:
        logger.info(f"Found existing {tpr} in {cwd}. Skipping.")
        return Result.SKIP


    sh(f"cp {parent_job}.top {job}.top")

    gro = f"{job}-start.gro"
    ts = env[f"{parent_job}-config-times"]
    ds = env[f"{parent_job}-config-ds"]
    ts_ds = sorted([(t, d) for t, d in zip(ts, ds)], key=lambda x: float(x[1]))
    choice = None
    previous = (None, None)
    for t, d in ts_ds:
        if float(d) >= TI_TARGET_DISTANCE:
            # use the time that is just below the target distance
            choice = previous
            break
        previous = (t, d)

    if choice is None:
        raise ValueError(f"Could not find a distance greater than {TI_TARGET_DISTANCE}")
    t, d = choice
    logger.info(f"Using configuration at time {t} with distance {d}.")
    sh(f"cp {parent_job}-conf-{t}-start.gro {gro}")
    env["wetbreak-start-t"] = t
    env["wetbreak-start-d"] = d
    write_env(env)

    charge = str(int(env[f"charge"]) - 1)
    write_qm_index(job, force)
    generate_qm_reference(charge=charge, job=job, force=force)

    basis_set = env.get("basis_set")
    if basis_set is None:
        raise ValueError("No basis set found in env.json")
    xc_functional = env.get("xc_functional", "PBE")
    cp2k_inp = f"{job}.inp"

    cp2k_reference_to_inp(
        in_reference=f"{job}-qm-reference_cp2k.inp",
        in_template=None,
        out_cp2k_inp=cp2k_inp,
        job=job,
        basis_set=basis_set,
        xc_functional=xc_functional,
    )

    system = cwd.split("/")[-2]
    external_force = env[f"{system}_external_force"]
    rotation_restraints = get_rotation_restraints(ASSETS, system)
    fill_template(
        Path(f"{ASSETS}/{job}.template.mdp"),
        Path(f"{job}.mdp"),
        CHARGE=charge,
        EXTERNAL_FORCE=external_force,
        ROTATION_RESTRAINTS=rotation_restraints,
        PREFIX=job,
    )
    sh(
        f"gmx grompp -n {job}.ndx -f {job}.mdp -c {gro} -r {gro} -qmi {cp2k_inp} -p {job}.top -o {tpr} -maxwarn 2"
    )
    logger.info(f"Generated {tpr} for {job} in {cwd}")
    return Result.OK




def update_local_env(global_env: dict) -> Result:
    """Update the local env.json with values from the global env.json.

    This was used only during iteration on the workflow and parameters.
    """
    local_env = read_env()
    if local_env is None:
        log_skipped("env.json", "update_local_env")
        return Result.SKIP
    for k, v in global_env.items():
        if k in local_env.keys():
            if local_env[k] != v:
                logger.info(f"Updating {k} from {local_env[k]} to {v}")
                confirmed = input("Press y to confirm.") == "y"
                if confirmed:
                    local_env[k] = v
        else:
            logger.info(f"Adding {k} from global_env of value {v} to local_env")
            local_env[k] = v

    write_env(local_env)

    return Result.OK

