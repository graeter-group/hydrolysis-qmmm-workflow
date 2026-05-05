import logging
import os
import shutil
from glob import glob
from itertools import product, repeat
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import hydrolysis.operations as op
import hydrolysis.utils as ut
from hydrolysis import mdatools
from hydrolysis.settings import N_FRAMES, derive_env_dirs, get_parameters, ASSETS
from hydrolysis.utils import (
    Result,
    check_logfiles_for_errors,
    link,
    log_skipped,
    log_waiting,
    lrange,
    pushd,
    read_env,
    sh,
    slurm_dispatch_systems,
    write,
    write_env,
)

logger = logging.getLogger("qm")


def setup():
    """
    Setup a logger that always writes to a logfile in the current working directory
    """
    logger.handlers = []
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("workflow.log")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.formatter = formatter
    file_handler.setLevel(logging.INFO)
    stream_handler.formatter = formatter
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # Mute others, e.g. tensorflow, matplotlib
    logging.getLogger().setLevel(logging.CRITICAL)

    logger.info("Starting workflow")


def setup_envs(force: bool = False, init: bool=False) -> list[dict]:
    envs = []
    for parts in product(*get_parameters()):
        d = {}
        for part in parts:
            d.update(part)
        d = derive_env_dirs(d)
        envs.append(d)

    if not init:
        return envs

    logger.info("Setup environments")
    env_dir = Path("./envs")
    if env_dir.exists() and not force:
        logger.info(f"skipping writing envs to disk.")
        return envs
    if not env_dir.exists():
        os.mkdir(env_dir)

    for i, env in enumerate(envs):
        main_dir = env["main_dir"]
        if not Path(main_dir).exists():
            os.mkdir(main_dir)
        dir = env["main_dir"] + "/envs"
        if not Path(dir).exists():
            os.mkdir(dir)
        write_env(env, dir + f"/env_{i}.json")
        write_env(env, env_dir / f"env_{main_dir}_{i}.json")

    return envs


def setup_chains(env: dict, force: bool = False):
    """Setup chains

    Setup simulation boxes for the single chains and the triplehelix.

    Outline of steps:
    mkdir; pdb2gmx; bo; solvate; neutralize with 0.15 molar NaCal;
    minimize; nvt; npt; make_ndx

    Parameters
    ----------
    env
        Environment dictionary instanziated from the settings
        in settings.py.
    force
        If True, force the setup of the chains even if they
        already exist.
    """
    cwd = env["main_dir"]

    logger.info("Setup directories")

    with pushd(cwd):
        for asset in [
            "amber99sb-star-ildnp.ff",
            "npt.mdp",
            "nvt.mdp",
            "ions.mdp",
            "minim.mdp",
            "_modules.sh",
            "chain-*.pdb",
            "triple.pdb",
            "jobscript.sh",
            "local-job.sh",
        ]:
            if Path(asset).exists() and not force:
                log_skipped(cwd, "setup chains")
                return
            link(f"{ASSETS}/{asset}")

        chains = ["triple"] + [f"chain-{x}" for x in lrange("a", "c")]
        for chain in tqdm(chains):
            op.setup_chain(chain=chain, force=force)


def equilibrate_chains(env: dict, local: bool = False, force: bool = False):
    """
    See equilibrate.template.mdp
    """
    force_dir = env["force_dir"]
    if not Path(force_dir).exists():
        os.mkdir(force_dir)

    logger.info(f"Equilibrate chains under external force.")
    with pushd(force_dir):
        link(f"{ASSETS}/amber99sb-star-ildnp.ff")
        link(f"{ASSETS}/_modules.sh")
        link(f"{ASSETS}/local-job.sh")
        link(f"{ASSETS}/jobscript.sh")
        if not Path("env.json").exists() and not force:
            force_env = env.copy()
            write_env(force_env)
        chains = ["triple"] + [f"chain-{x}" for x in lrange("a", "c")]
        for chain in tqdm(chains):
            op.equilibrate_chain(chain=chain, local=local, force=force)


def analyse_equilibrate_chains(env: dict, force=False):
    """
    See equilibrate.template.mdp
    """
    force_dir = env["force_dir"]
    with pushd(force_dir):
        chains = ["triple"] + [f"chain-{x}" for x in lrange("a", "c")]

        for chain in tqdm(chains):
            op.analyse_equilibrated_chain(chain=chain, force=force)
            op.export_equilibrated_chain_frames(chain=chain, n=N_FRAMES, force=force)


def setup_system_dir(env, system, frames, force=False, reset=False):
    """
    Prepares system_dir (single/triple)
    """
    bond_dir = env[f"peptide_bond_dir"]
    with pushd(bond_dir):
        bond_env = read_env()
        if not bond_env:
            log_waiting("env.json", f"in {bond_dir}")
            return
        if system == "single":
            bond_env.update({k.removeprefix("single_"): v for k, v in bond_env.items()})
            name = f'chain-{bond_env["chain"]}'
        elif system == "triple":
            bond_env.update({k.removeprefix("triple_"): v for k, v in bond_env.items()})
            name = "triple"
        else:
            raise ValueError(f"unknown dir {system}")

    system_dir = env[f"{system}_dir"]
    if not Path(system_dir).exists():
        os.mkdir(system_dir)
    with pushd(system_dir):
        system_env = bond_env.copy()
        system_env = mdatools.get_initial_qm_atoms(
            system_env, system, f"../../{name}-eq.gro"
        )
        write_env(system_env)

        for frame in tqdm(frames):
            frame_env = system_env.copy()
            frame_dir = f"frame-{frame}"
            if Path(frame_dir).exists() and not force:
                log_skipped(frame_dir, "setup_system_dir")
                continue
            if Path(frame_dir).exists() and force and reset:
                shutil.rmtree(frame_dir)
            if not Path(frame_dir).exists():
                os.mkdir(frame_dir)

            with pushd(frame_dir):
                dir = frame_env[f"{system}_dir"] + "/" + frame_dir
                frame_env["frame_dir"] = dir
                frame_env["dir"] = dir
                top = f"../../../{name}-topol.top"
                gro = f"../../../{name}-frames/{name}-frame-{frame}.gro"
                gro = shutil.copy(gro, "eq.gro")
                top = shutil.copy(top, "eq.top")

                link(f"{ASSETS}/local-job.sh")
                link(f"{ASSETS}/jobscript.sh")
                link(f"{ASSETS}/_modules.sh")
                link(f"{ASSETS}/amber99sb-star-ildnp.ff")

                logger.info(f"setup qm dir: {dir}.")
                write_env(frame_env)


def setup_qm(env, force=False):
    """
    needs equilibrated systems
    """
    bond_dir = env[f"peptide_bond_dir"]
    force_dir = env["force_dir"]
    logger.info(f"Setup QM for {force_dir}")
    with pushd(force_dir):
        for f in ["triple"] + [f"chain-{x}" for x in lrange("a", "c")]:
            depends = f"{f}-eq.gro"
            if not Path(depends).exists():
                log_waiting(depends, f"setup_qm in {force_dir}")
                return
        frames = [
            x.removesuffix(".gro").split("-")[-1] for x in glob("triple-frames/*.gro")
        ]
        if not frames:
            log_waiting("export_frames", f"setup_qm in {force_dir}")
            return

    if not Path(bond_dir).exists():
        os.mkdir(bond_dir)
    with pushd(bond_dir):
        bond_env = env.copy()
        if not Path("env.json").exists() or force:
            bond_env["frames"] = frames
            write_env(bond_env)
            op.match_triple_to_single_chain_residues(force=force)

    for system in ["single", "triple"]:
        setup_system_dir(env, system, frames, force)


def setup_wethyd_frame(args):
    frame, system_dir, force = args
    frame_dir = f"{system_dir}/frame-{frame}"
    with pushd(frame_dir):
        op.setup_wethyd(force=force)


def setup_wethyd_warmup_frame(args):
    frame, system_dir, force = args
    frame_dir = f"{system_dir}/frame-{frame}"
    with pushd(frame_dir):
        op.setup_wethyd_warmup(force=force)


def setup_wethyd_after_warmup_frame(args):
    frame, system_dir, force = args
    frame_dir = f"{system_dir}/frame-{frame}"
    with pushd(frame_dir):
        op.setup_wethyd_after_warmup(force=force)


def setup_wethyds_warmup(env, parallel=True, force=False):
    """
    Depends on st.setup_qm.
    """
    job = "wethyd-warmup"
    for system in ["single", "triple"]:
        system_dir = env[f"{system}_dir"]
        with pushd(system_dir):
            logger.info(f"cwd: {system_dir}")
            logger.info(f"step: {job}")
            system_env = read_env()

            if not system_env:
                log_waiting("setup_qm", f"setup_hyd")
                continue
            dep = "eq"
            depends = f"{dep}.gro"
            if not depends:
                log_waiting("eq", f"setup_hyd")
                continue
        frames = system_env["frames"]
        frames.sort(key=id)

        for frame in tqdm(frames, desc="frames"):
            frame_dir = f"{system_dir}/frame-{frame}"
            with pushd(frame_dir):
                op.update_local_env(env)

        if parallel:
            with Pool(min(len(frames), 10)) as p:
                p.map(
                    setup_wethyd_warmup_frame,
                    zip(frames, repeat(system_dir), repeat(force)),
                )
        else:
            for frame in tqdm(frames, desc="frames"):
                frame_dir = f"{system_dir}/frame-{frame}"
                with pushd(frame_dir):
                    op.update_local_env(env)
                    op.setup_wethyd_warmup(force=force)


def setup_wethyds_after_warmup(env, parallel=True, force=False):
    """
    Depends on st.setup_qm.
    """
    job = "wethyd"
    for system in ["single", "triple"]:
        system_dir = env[f"{system}_dir"]
        with pushd(system_dir):
            logger.info(f"cwd: {system_dir}")
            logger.info(f"step: {job}")
            system_env = read_env()

            if not system_env:
                log_waiting("setup_qm", f"setup_hyd")
                continue
            dep = "eq"
            depends = f"{dep}.gro"
            if not depends:
                log_waiting("eq", f"setup_hyd")
                continue
        frames = system_env["frames"]
        frames.sort(key=id)

        if parallel:
            with Pool(min(len(frames), 10)) as p:
                p.map(
                    setup_wethyd_after_warmup_frame,
                    zip(frames, repeat(system_dir), repeat(force)),
                )
        else:
            for frame in tqdm(frames, desc="frames"):
                frame_dir = f"{system_dir}/frame-{frame}"
                with pushd(frame_dir):
                    op.setup_wethyd_after_warmup(force=force)


def setup_wethyds(env, parallel=True, force=False):
    """
    Depends on st.setup_qm.
    """
    job = "wethyd"
    for system in ["single", "triple"]:
        system_dir = env[f"{system}_dir"]
        with pushd(system_dir):
            logger.info(f"cwd: {system_dir}")
            logger.info(f"step: {job}")
            system_env = read_env()

            if not system_env:
                log_waiting("setup_qm", f"setup_hyd")
                continue
            dep = "eq"
            depends = f"{dep}.gro"
            if not depends:
                log_waiting("eq", f"setup_hyd")
                continue
        frames = system_env["frames"]
        frames.sort(key=id)

        for frame in tqdm(frames, desc="frames"):
            frame_dir = f"{system_dir}/frame-{frame}"
            with pushd(frame_dir):
                op.update_local_env(env)

        if parallel:
            with Pool(min(len(frames), 10)) as p:
                p.map(
                    setup_wethyd_frame,
                    zip(frames, repeat(system_dir), repeat(force)),
                )
        else:
            for frame in tqdm(frames, desc="frames"):
                frame_dir = f"{system_dir}/frame-{frame}"
                with pushd(frame_dir):
                    op.update_local_env(env)
                    op.setup_wethyd(force=force)


def setup_wethyd_recovery(env, force=False):
    """
    Depends on st.setup_qm.
    This turned out to not work particularly well.
    More diverse starting points are more promising than
    restarting with minor tweaks with different velocities etc.
    """
    raise DeprecationWarning("This approach is deprecated")
    job = "wethyd"
    for system in ["single", "triple"]:
        system_dir = env[f"{system}_dir"]
        with pushd(system_dir):
            logger.info(f"cwd: {system_dir}")
            logger.info(f"step: {job}")
            system_env = read_env()
            if not system_env:
                log_waiting("setup_qm", f"setup_hyd")
                continue
            dep = "eq"
            depends = f"{dep}.gro"
            if not depends:
                log_waiting("eq", f"setup_hyd")
                continue
        frames = system_env["frames"]
        frames.sort(key=id)
        for frame in frames:
            frame_dir = f"{system_dir}/frame-{frame}"
            with pushd(frame_dir):
                op.setup_wethyd_recovery(force=force)


def start_wethyd_warmups(
    env: dict,
    force: bool = False,
    append: bool = False,
    clear_restart_files: bool = False,
):
    job = "wethyd-warmup"
    slurm_dispatch_systems(
        env=env,
        job=job,
        force=force,
        append=append,
        clear_restart_files=clear_restart_files,
        max_time=0.0198,  # one step less than target time in ps (0.02 ps)
    )


def start_wethyds(
    env: dict,
    force: bool = False,
    append: bool = False,
    clear_restart_files: bool = False,
):
    job = "wethyd"
    slurm_dispatch_systems(
        env=env,
        job=job,
        force=force,
        append=append,
        clear_restart_files=clear_restart_files,
    )


def check_wethyd_warmup_outcomes(env):
    job = "wethyd-warmup"
    run = env["main_dir"]
    basis = env["basis_set"]
    xc_functional = env["xc_functional"]
    errors = []
    for system in ["single", "triple"]:
        system_dir = env[f"{system}_dir"]
        with pushd(system_dir):
            logger.info(f"cwd: {system_dir}")
            logger.info(f"step: {job}")
            system_env = read_env()

            if not system_env:
                log_waiting("setup_qm", f"setup_hyd")
                continue
            dep = "eq"
            depends = f"{dep}.gro"
            if not depends:
                log_waiting("eq", f"setup_hyd")
                continue
        frames = system_env["frames"]
        frames.sort(key=id)
        for frame in tqdm(frames, desc="frames"):
            frame_dir = f"{system_dir}/frame-{frame}"
            with pushd(frame_dir):
                logcheck = check_logfiles_for_errors(job)
                error_msg = logcheck.error_msg
                found_error = logcheck.found_error
                error_type = logcheck.error_type
                latest_time = logcheck.latest_time
                latest_distance = logcheck.latest_distance
                if type(error_msg) is list:
                    msg = r"\n".join(error_msg)
                else:
                    msg = error_msg
                errors.append(
                    {
                        "job": job,
                        "run": run,
                        "basis": basis,
                        "xc": xc_functional,
                        "external_force": env["single_external_force"],
                        "ix_c": env["triple_ix_c_carbonyl"],
                        "ix_n": env["triple_ix_n_peptide"],
                        "system": system,
                        "frame": frame,
                        "error": found_error,
                        "msg": msg,
                        "type": error_type,
                        "time": latest_time,
                        "d": latest_distance,
                    }
                )
    return errors


def check_wethyd_outcomes(env):
    job = "wethyd"
    run = env["main_dir"]
    basis = env["basis_set"]
    xc_functional = env["xc_functional"]
    errors = []
    for system in ["single", "triple"]:
        system_dir = env[f"{system}_dir"]
        with pushd(system_dir):
            logger.info(f"cwd: {system_dir}")
            logger.info(f"step: {job}")
            system_env = read_env()

            if not system_env:
                log_waiting("setup_qm", f"setup_hyd")
                continue
            dep = "eq"
            depends = f"{dep}.gro"
            if not depends:
                log_waiting("eq", f"setup_hyd")
                continue
        frames = system_env["frames"]
        frames.sort(key=id)
        for frame in tqdm(frames, desc="frames"):
            frame_dir = f"{system_dir}/frame-{frame}"
            with pushd(frame_dir):
                logcheck = check_logfiles_for_errors(job)
                error_msg = logcheck.error_msg
                found_error = logcheck.found_error
                error_type = logcheck.error_type
                latest_time = logcheck.latest_time
                latest_distance = logcheck.latest_distance

                if type(error_msg) is list:
                    msg = r"\n".join(error_msg)
                else:
                    msg = error_msg
                errors.append(
                    {
                        "job": job,
                        "run": run,
                        "basis": basis,
                        "xc": xc_functional,
                        "external_force": env["single_external_force"],
                        "ix_c": env["triple_ix_c_carbonyl"],
                        "ix_n": env["triple_ix_n_peptide"],
                        "system": system,
                        "frame": frame,
                        "error": found_error,
                        "msg": msg,
                        "type": error_type,
                        "time": latest_time,
                        "d": latest_distance,
                    }
                )
    return errors


def check_wethyd_us_outcomes(env):
    parent_job = "wethyd"
    run = env["main_dir"]
    basis = env["basis_set"]
    errors = []
    for system in ["single", "triple"]:
        system_dir = env[f"{system}_dir"]
        system_env = read_env(system_dir)
        assert system_env
        frames = system_env["frames"]
        frames.sort(key=id)
        for frame in tqdm(frames, desc="frames"):
            frame_dir = f"{system_dir}/frame-{frame}"
            with pushd(frame_dir):
                cwd = os.getcwd()
                frame_env = read_env()
                assert frame_env
                ts = frame_env.get(f"{parent_job}-config-times")
                if ts is None:
                    logger.warning(f"no times found in {cwd}")
                    continue
                for t in ts:
                    job = f"{parent_job}-conf-{t}"
                    logcheck = check_logfiles_for_errors(job)
                    error_msg = logcheck.error_msg
                    found_error = logcheck.found_error
                    error_type = logcheck.error_type
                    latest_time = logcheck.latest_time
                    latest_distance = logcheck.latest_distance

                    if type(error_msg) is list:
                        msg = r"\n".join(error_msg)
                    else:
                        msg = error_msg
                    errors.append(
                        {
                            "cwd": cwd,
                            "job": job,
                            "run": run,
                            "basis": basis,
                            "external_force": frame_env["single_external_force"],
                            "ix_c": frame_env["triple_ix_c_carbonyl"],
                            "ix_n": frame_env["triple_ix_n_peptide"],
                            "system": system,
                            "frame": frame,
                            "t": t,
                            "error": found_error,
                            "msg": msg,
                            "type": error_type,
                            "time": latest_time,
                            "d": latest_distance,
                        }
                    )
    return errors


def start_wethyd_recovery(env, force=False, append=False):
    job = "wethyd-recovery"
    slurm_dispatch_systems(env=env, job=job, force=force, append=append)


def analyse_job_in_dir(args: tuple[str, str, list[str], bool]):
    dir, job, analysis_types, force = args
    with pushd(dir):
        op.analyse_job(job=job, analysis_types=analysis_types, force=force)


def analyse_wethyds(
    envs: list[dict],
    analysis_types: list[str] = ["distances", "energy", "center"],
    force: bool = False,
):
    job = "wethyd"
    dirs = []
    for env in envs:
        for system in ["single", "triple"]:
            with pushd(env[f"{system}_dir"]):
                system_env = read_env()
                assert system_env
                frames = system_env["frames"]
                frame_dirs = [
                    env[f"{system}_dir"] + f"/frame-{frame}" for frame in frames
                ]
                dirs += frame_dirs

    with Pool(10) as p:
        p.map(
            analyse_job_in_dir,
            zip(dirs, repeat(job), repeat(analysis_types), repeat(force)),
        )

def setup_us_in_dir(args: tuple[str, str, bool, bool]):
    dir, parent_job, force, extend = args
    with pushd(dir):
        op.get_umbrella_window_times(
            parent_job=parent_job, force=force, extend=extend
        )
        op.extract_umbrella_configs(parent_job=parent_job, force=force)
        op.setup_umbrella_sampling(parent_job=parent_job, force=force)


def setup_wethyd_us(
    choices: pd.DataFrame,
    force: bool = False,
    extend: bool = False,
    parallel: bool = True,
):
    parent_job = "wethyd"
    if parallel:
        with Pool(min(len(choices), 10)) as p:
            p.map(
                setup_us_in_dir,
                zip(
                    choices.cwd,
                    repeat(parent_job),
                    repeat(force),
                    repeat(extend),
                ),
            )
    else:
        for l in tqdm(choices.itertuples(index=False), desc="choice"):
            cwd = l.cwd  # type:ignore
            with pushd(cwd):
                logger.info(f"Setup wethyd US in {cwd}")
                op.get_umbrella_window_times(
                    parent_job=parent_job, force=force, extend=extend
                )
                op.extract_umbrella_configs(parent_job=parent_job, force=force)
                op.setup_umbrella_sampling(parent_job=parent_job, force=force)


def start_wethyd_us(
    choices: pd.DataFrame, force: bool = False, append=False
) -> Result:
    parent_job = "wethyd"
    for l in tqdm(choices.itertuples(index=False), desc="choice"):
        cwd = l.cwd  # type:ignore
        env = read_env(cwd)
        if not env:
            log_waiting("env.json", f"in {cwd}")
            return Result.WAIT
        ts = env.get(f"{parent_job}-config-times")
        if not ts:
            log_waiting("extract umbrella configs", f"in {cwd} for {parent_job}")
            return Result.WAIT
        for t in ts:
            conf = f"{parent_job}-conf-{t}"
            op.slurm_dispatch(
                cwd=cwd,
                job=conf,
                name=None,
                force=force,
                append=append,
                # at least one dt less than target time in ps
                # max_time=4.999,
                # setting lower to prioritize getting all jobs 
                # to a couple of ps over finishing longer ones
                max_time=0.998,
            )
    return Result.OK


def analyse_wethyd_us(choices: pd.DataFrame, to_discard: pd.DataFrame|None = None, force: bool = False):
    for l in choices.itertuples():
        cwd = l.cwd  # type: ignore
        with pushd(cwd):
            logger.info(f"Analyse wethyd US in {cwd}")
            op.analyse_wethyd_us(force=force, to_discard=to_discard)


def analyse_wethyd_us_energies(choices: pd.DataFrame, force=False):
    for l in choices.itertuples():
        cwd = l.cwd # type: ignore
        with pushd(cwd):
            logger.info(f"Analyse wethyd US in {cwd}")
            op.analyse_wethyd_us_energies(force=force)


def analyse_wethyd_us_combined(choices: pd.DataFrame, to_discard: pd.DataFrame|None = None, force: bool=False):
    op.analyse_combined_us(parent_job="wethyd", choices=choices, to_discard=to_discard, force=force)


def analyse_ti_proton_distances(choices: pd.DataFrame, force=False):
    for l in choices.itertuples():
        cwd = l.cwd  # type:ignore
        with pushd(cwd):
            logger.info(f"Analyse wethyd in {cwd}")
            op.analyse_ti_proton_distances(force=force)

def analyse_ti_protonation(choices: pd.DataFrame, force=False):
    for l in choices.itertuples():
        cwd = l.cwd  # type:ignore
        with pushd(cwd):
            logger.info(f"Analyse ti protonation in {cwd}")
            op.analyse_ti_protonation(force=force)

def analyse_ti_stability(choices: pd.DataFrame, force=False):
    for l in choices.itertuples():
        cwd = l.cwd  # type:ignore
        with pushd(cwd):
            logger.info(f"Analyse ti stability in {cwd}")
            op.analyse_ti_stability(force=force)

def analyse_wetbreak_us(choices: pd.DataFrame, force=False):
    for l in choices.itertuples():
        cwd = l.cwd  # type:ignore
        with pushd(cwd):
            logger.info(f"Analyse wetbreak US in {cwd}")
            op.analyse_wetbreak_us(force=force)

def analyse_wetbreak_us_combined(choices: pd.DataFrame, to_discard: pd.DataFrame|None = None, force: bool=False):
    op.analyse_combined_us(parent_job="wetbreak", choices=choices, to_discard=to_discard, force=force)

def analyse_break_ti_distances(choices: pd.DataFrame, force=False):
    for l in choices.itertuples():
        cwd = l.cwd  # type:ignore
        with pushd(cwd):
            logger.info(f"Analyse ti stability of break in {cwd}")
            op.analyse_break_ti_distances(force=force)

def analyse_break_proton_distances(choices: pd.DataFrame, force=False):
    for l in choices.itertuples():
        cwd = l.cwd  # type:ignore
        with pushd(cwd):
            logger.info(f"Analyse protonation states of break in {cwd}")
            op.analyse_break_proton_distances(force=force)


def center_vis(env, system, frame, job, force=False):
    cwd = env[f"{system}_dir"] + f"/frame-{frame}"
    with pushd(cwd):
        op.center_vis(job, force)

def center_vis_in(cwd: str|Path, job: str, force: bool = False, **kwargs):
    with pushd(cwd):
        op.center_vis(job, force, **kwargs)

def open_vis(env, system, frame, job, glxfix=True):
    cwd = env[f"{system}_dir"] + f"/frame-{frame}"
    vmdfile = f"{job}.vmd"
    if glxfix:
        fix = "__GLX_VENDOR_LIBRARY_NAME=mesa "
    else:
        fix = ""
    with pushd(cwd):
        sh(f"{fix}vmd -e {vmdfile}")


def center_wethyd_us_xtcs(env, force=False):
    parent_job = "wethyd"
    for system in ["single", "triple"]:
        system_dir = env[f"{system}_dir"]
        with pushd(system_dir):
            system_env = read_env(system_dir)
            assert system_env
            frames = system_env["frames"]
        for frame in frames:
            frame_dir = f"{system_dir}/frame-{frame}"
            with pushd(frame_dir):
                frame_env = read_env(frame_dir)
                assert frame_env
                ts = frame_env.get(f"{parent_job}-config-times")
                assert ts
                for t in ts:
                    job = f"{parent_job}-conf-{t}"
                    op.center_xtc(job, force)


def setup_wetbreak(
    successful_combinations: pd.DataFrame, force: bool = False, hard_reset: bool = False
):
    parent_job = "wethyd"
    for l in successful_combinations.itertuples(index=False):
        cwd = l.cwd  # type:ignore
        with pushd(cwd):
            parent_trajectory = f"{parent_job}.xtc"
            if not Path(parent_trajectory).exists():
                return
            logger.info(f"Setup wetbreak in {cwd}")
            op.setup_wetbreak(force=force, hard_reset=hard_reset)

def setup_wetbreak_restrained(
    successful_combinations: pd.DataFrame, force: bool = False, hard_reset: bool = False
):
    parent_job = "wethyd"
    for l in successful_combinations.itertuples(index=False):
        cwd = l.cwd  # type:ignore
        with pushd(cwd):
            parent_trajectory = f"{parent_job}.xtc"
            if not Path(parent_trajectory).exists():
                return
            logger.info(f"Setup wetbreak in {cwd}")
            op.setup_wetbreak_restrained(force=force, hard_reset=hard_reset)



def start_wetbreak(
    successful_combinations: pd.DataFrame, force: bool = False, append: bool = False
):
    job = "wetbreak"
    for l in successful_combinations.itertuples(index=False):
        cwd = l.cwd  # type:ignore
        op.slurm_dispatch(cwd=cwd, job=job, name=None, force=force, append=append)


def setup_wetbreak_us(successful_combinations: pd.DataFrame, force: bool = False):
    parent_job = "wetbreak"
    for l in successful_combinations.itertuples(index=False):
        cwd = f"{l.run}/f-{l.external_force}/ixs-{l.ix_c}-{l.ix_n}/{l.system}/frame-{l.frame}"  # type: ignore
        with pushd(cwd):
            logger.info(f"Setup wetbreak US in {cwd}")
            op.get_umbrella_window_times(
                parent_job, cutoff=0.4, further=10, closer=30, extend=True, force=force
            )
            op.extract_umbrella_configs(parent_job, force=force)
            op.setup_umbrella_sampling(parent_job, force=force)


def start_wetbreak_us(successful_combinations: pd.DataFrame, force: bool = False, append: bool = False):
    parent_job = "wetbreak"
    for l in successful_combinations.itertuples(index=False):
        cwd = f"{l.run}/f-{l.external_force}/ixs-{l.ix_c}-{l.ix_n}/{l.system}/frame-{l.frame}"  # type: ignore
        env = read_env(cwd)
        if not env:
            log_waiting("env.json", f"in {cwd}")
            return
        ts = env.get(f"{parent_job}-config-times")
        if not ts:
            log_waiting("extract umbrella coonfigs", f"in {cwd} for {parent_job}")
            return
        for t in ts:
            conf = f"{parent_job}-conf-{t}"
            op.slurm_dispatch(cwd=cwd, job=conf, name=None, force=force, append=append)
