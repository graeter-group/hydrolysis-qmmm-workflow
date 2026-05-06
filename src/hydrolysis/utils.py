import enum
import json
import logging
import os
import pprint
import re
import shutil
import subprocess as sp
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import reduce
from glob import glob
from operator import add
from pathlib import Path
from string import Template
from typing import Any, Callable, Optional, TypeVar, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cp2k_input_tools.generator import CP2KInputGenerator
from cp2k_input_tools.parser import CP2KInputParserSimplified
from MDAnalysis.coordinates.H5MD import mda
from numpy import int64

from hydrolysis.constants import NO_WATER_FOUND

CLUSTER = "cascade-login.h-its.org"

T = TypeVar("T")


class Result(Enum):
    OK = auto()
    WAIT = auto()
    SKIP = auto()
    ERROR = auto()
    WARN = auto()


logger = logging.getLogger("qm")


@contextmanager
def pushd(path):
    prev = os.getcwd()
    logger.info(f"Pushing to {path}")
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def write_env(env: dict, p: str | Path = "env.json"):
    d = {}
    for k, v in env.items():
        t = type(v)
        if t == int64 or t == int:
            v = str(v)
        d[k] = v

    logger.info(f"Writing environment to {p}")
    with open(p, "w") as f:
        json.dump(d, f)


def read_env(p: Path | str | None = None) -> dict[str, Any] | None:
    if p is None:
        p = Path("env.json")
    p = Path(p)
    if p.is_dir():
        p = Path(p) / Path("env.json")
    logger.info(f"Reading environment {p}")
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as _:
        logger.warning(f"Environment not found in {p}")
        return None


def log_skipped(target, step):
    logger.info(f"{target} already exists. Skipping step: {step}.")


def log_waiting(depend, step):
    logger.info(f"Dependency {depend} doesn't exist. Skipping step: {step}.")


def pp(*args):
    pprint.pprint(*args)


def sh(
    cmd: str,
    env: dict | None = None,
    cwd=".",
    quiet: int = 2,
    capture: bool = False,
    check: bool = True,
):
    stdout = None
    stderr = None
    if capture:
        stdout = sp.PIPE
        stderr = sp.PIPE
    if quiet >= 1:
        stdout = sp.DEVNULL
    if quiet >= 2:
        stderr = sp.DEVNULL
    return sp.run(
        args=cmd,
        env=env,
        cwd=cwd,
        shell=True,
        check=check,
        text=True,
        stdout=stdout,
        stderr=stderr,
    )


def shget(cmd, env=None, cwd=".") -> str:
    return sh(cmd=cmd, env=env, cwd=cwd, quiet=0, capture=True).stdout


def write(s: Union[list[str], str], p: Union[str, Path]):
    with open(p, "w") as f:
        f.writelines(s)


def read(p: str):
    with open(p, "r") as f:
        return [l.strip() for l in f.readlines()]


def lrange(x, y):
    return [chr(i) for i in range(ord(x), ord(y) + 1)]


def ln(src: str, dst: str):
    sh(f"ln -srf {src} {dst}")


def ln_to(src: str, dst: str):
    sh(f"ln -srf {src} -t {dst}")


def link(src: str):
    logger.info(f"Linking {src} to .")
    ln_to(src, ".")


def clean_artifacts(cwd, exclude: Optional[list] = None):
    """
    Remove all backup files of the form #*# and slurm-*
    """
    if Path(cwd).exists():
        with pushd(cwd):
            fs = reduce(
                add,
                [
                    glob("#*#"),
                    glob("slurm-*.out"),
                    glob("*.nfs*"),
                    glob("*.info"),
                    glob("*.wfn.bak-?"),
                    glob("frame-*/#*#"),
                    glob("frame-*/slurm-*.out"),
                    glob("frame-*/*.nfs*"),
                    glob("frame-*/*.info"),
                    glob("frame-*/*.wfn.bak-?"),
                ],
            )
            for f in fs:
                m = re.match(r"((frame-\d+/)?\w+(-recovery)?)", f)
                jobname = f
                if m is not None:
                    jobname = m.group(0)
                check = cwd + "/" + jobname
                if check in exclude:
                    logger.info(f"Skipping {f}")
                    continue
                logger.info(f"Removing {f}")
                os.remove(f)


def clean_all(env, exclude: Optional[list] = None):
    """
    Remove all backup files of the form #*# and slurm-*
    """
    for k, v in env.items():
        if k.endswith("_dir"):
            clean_artifacts(v, exclude)


def get_peptide_bond_ixs(env):
    """
    Get the C and N of the peptide bond and the amino acid sequence for each of the peptides in the triplehelix.
    """
    u = mda.Universe(f"{env['main_dir']}/triple-box.gro")
    peptide_bond_atoms = u.select_atoms("name C or name N")
    df = pd.DataFrame(
        {
            "ix_c": peptide_bond_atoms.ix[::2],
            "res": peptide_bond_atoms.resnames[::2],
            "resid": peptide_bond_atoms.resids[::2],
        }
    )
    df["ix_n"] = df["ix_c"] + 2
    df["start"] = df["res"] == "ACE"
    df["chain"] = df["start"].astype(int).cumsum()
    del df["start"]
    df["chain"] = df["chain"].apply(lambda x: chr(x + 96))
    return df


def fill_template(path_template: Path, path_output: Path, **kwargs):
    if not path_template.exists():
        raise FileNotFoundError(path_template)

    if path_output == "":
        raise ValueError("Output path is empty")

    with open(path_template, "r") as f:
        template = Template(f.read())

    with open(path_output, "w") as f:
        f.write(template.safe_substitute(kwargs))


def cp2k_reference_to_inp(
    in_reference: str,
    in_template: str | None,
    out_cp2k_inp: str,
    job: str,
    basis_set: str = "DZVP-MOLOPT-GTH",
    xc_functional: str = "PBE",
):
    logger.info(f"Converting cp2k reference {in_reference} to template {out_cp2k_inp}")
    parser = CP2KInputParserSimplified()
    generator = CP2KInputGenerator()

    with open(in_reference, "r") as f:
        reference = parser.parse(f)

    if in_template is not None:
        raise DeprecationWarning("in_template is not used anymore")

    reference["global"]["project_name"] = job
    reference["force_eval"]["subsys"]["topology"]["coord_file_name"] = f"{job}_cp2k.pdb"

    reference["force_eval"]["dft"]["qs"]["method"] = "GPW"
    reference["force_eval"]["dft"]["qs"]["eps_default"] = 1.0e-10  # default 1.0e-10

    reference["force_eval"]["dft"]["scf"]["scf_guess"] = "RESTART"
    reference["force_eval"]["dft"]["scf"]["eps_scf"] = 1.0e-6  # default 1.0e-8
    reference["force_eval"]["dft"]["scf"]["max_scf"] = 20
    reference["force_eval"]["dft"]["scf"]["outer_scf"]["max_scf"] = 20
    reference["force_eval"]["dft"]["scf"]["outer_scf"][
        "eps_scf"
    ] = 1.0e-6  # default 1.0e-8

    if xc_functional == "PBE":
        reference["force_eval"]["dft"]["xc"] = {
            "DENSITY_CUTOFF": 1.0e-12,
            "GRADIENT_CUTOFF": 1.0e-12,
            "TAU_CUTOFF": 1.0e-12,
            "XC_FUNCTIONAL": {"PBE": {}},
            "vdW_POTENTIAL": {
                "DISPERSION_FUNCTIONAL": "PAIR_POTENTIAL",
                "PAIR_POTENTIAL": {
                    "TYPE": "DFTD3",
                    "PARAMETER_FILE_NAME": "/hits/fast/mbm/buhrjk/sw/cp2k/data/dftd3.dat",
                    "REFERENCE_FUNCTIONAL": "PBE",
                },
            },
        }

        for atom in reference["force_eval"]["subsys"]["kind"].keys():
            if atom == "X":
                continue
            reference["force_eval"]["subsys"]["kind"][atom]["basis_set"] = basis_set
    elif xc_functional == "PBE0":
        raise NotImplementedError("PBE0 not implemented yet")
    elif xc_functional == "B3LYP":
        reference["force_eval"]["dft"]["xc"] = {
            "XC_FUNCTIONAL": {
                "LYP": {"SCALE_C": 0.81},
                "BECKE88": {"SCALE_X": 0.72},
                "VWN": {"FUNCTIONAL_TYPE": "VWN3", "SCALE_C": 0.19},
                "XALPHA": {"SCALE_X": 0.08},
            },
            "HF": {
                "FRACTION": 0.20,
                "Screening": {"EPS_SCHWARZ": 1.0e-10},
                "INTERACTION_POTENTIAL": {
                    "POTENTIAL_TYPE": "TRUNCATED",  # for condensed phase systems
                    "CUTOFF_RADIUS": 6.0,  # should be less than halve the cell
                    "T_C_G_DATA": "t_c_g.dat",  # data file needed with the truncated operator
                },
                "MEMORY": {"MAX_MEMORY": 1500},
            },
        }

        # add vdW dispersion correction
        reference["force_eval"]["dft"]["xc"]["vdW_POTENTIAL"] = {
            "DISPERSION_FUNCTIONAL": "PAIR_POTENTIAL",
            "PAIR_POTENTIAL": {
                "TYPE": "DFTD3",
                "PARAMETER_FILE_NAME": "dftd3.dat",
                "REFERENCE_FUNCTIONAL": "B3LYP",
            },
        }

        # add ADMM
        reference["force_eval"]["dft"]["auxiliary_density_matrix_method"] = {
            "METHOD": "BASIS_PROJECTION",
            "ADMM_PURIFICATION_METHOD": "MO_DIAG",
        }
        for atom in reference["force_eval"]["subsys"]["kind"].keys():
            if atom == "H":
                i = 1
            elif atom == "C":
                i = 4
            elif atom == "N":
                i = 5
            elif atom == "O":
                i = 6
            elif atom == "S":
                i = 6
            elif atom == "X":
                continue
            else:
                raise NotImplementedError(
                    f"Atom {atom} not implemented for potential selection."
                )
            reference["force_eval"]["subsys"]["kind"][atom]["basis_set"] = [
                basis_set,
                ("AUX_FIT", "SZV-MOLOPT-GTH"),
            ]
            reference["force_eval"]["subsys"]["kind"][atom][
                "potential"
            ] = f"GTH-BLYP-q{i}"

    reference["force_eval"]["subsys"]["topology"]["center_coordinates"] = {}

    with open(out_cp2k_inp, "w") as f:
        for l in generator.line_iter(reference):
            l = l.replace(".TRUE.", "TRUE").replace(".FALSE.", "FALSE")
            f.write(f"{l}\n")


class Error(Enum):
    GmxError = auto()
    Cp2kError = auto()
    SlurmError = auto()
    NoWaterError = auto()


@dataclass
class LogCheckResult:
    """Result of checking logfiles for errors.

    Attributes
    ----------
    found_error
        True if an error was found
    error_type
        The type of error found
    error_msg
        The error message
    latest_time
        The latest time in the pullx file
    latest_distance
        The latest distance in the pullx file
    """

    found_error: bool = False
    error_type: Error | None = None
    error_msg: str | None | list[str] = None
    latest_time: float = 0.0
    latest_distance: float | None = None


def check_logfiles_for_errors(
    job: str,
) -> LogCheckResult:
    error_type = None
    error_msg = None
    latest_time = 0.0
    latest_distance = None
    path_log = f"{job}.log"
    path_cp2kout = f"{job}_cp2k.out"
    path_pullx = f"{job}_pullx.xvg"

    if Path(NO_WATER_FOUND).exists():
        logger.info(f"No initial configuration found due to lack of water")
        return LogCheckResult(
            found_error=True,
            error_type=Error.NoWaterError,
            error_msg="no water found",
            latest_time=latest_time,
            latest_distance=latest_distance,
        )

    if not (Path(path_log).exists() and Path(path_cp2kout).exists()):
        logger.info(f"No logfile exists, not checking previous runs for errors")
        return LogCheckResult(
            found_error=False,
            error_type=error_type,
            error_msg=error_msg,
            latest_time=latest_time,
            latest_distance=latest_distance,
        )

    found_error = False
    gmx_log = shget(f"tail -n 50 {path_log}")
    cp2k_log = shget(f"tail -n 50 {path_cp2kout}")
    if Path(f"{job}-slurm.out").exists():
        slurm_out = shget(f"tail -n 50 {job}-slurm.out")
    else:
        slurm_out = ""
    if "ABORT" in cp2k_log:
        found_error = True
        logger.info(f"Found ABORT in {job}_cp2k.out")
        msg = shget(f"grep -B 4 -A 6 'ABORT' {job}_cp2k.out")
        logger.info(msg)
        error_msg = msg
        error_type = Error.Cp2kError
    elif "Fatal error:" in gmx_log:
        found_error = True
        logger.info(f"Found Fatal error in {job}.log")
        msg = shget(f"grep -B 0 -A 2 'Fatal error' {job}.log")
        logger.info(msg)
        error_msg = msg
        error_type = Error.GmxError
    elif "Fatal error" in slurm_out:
        found_error = True
        logger.info(f"Found Fatal error in {job}-slurm.out")
        msg = shget(f"grep -B 2 -A 6 'Fatal error' {job}-slurm.out")
        logger.info(msg)
        error_msg = msg
        error_type = Error.SlurmError

    if Path(path_pullx).exists():
        pull_tail = shget(f"tail -n 50 {path_pullx}")
        if pull_tail != "" and pull_tail != "\n" and type(pull_tail) is str:
            # use second to last line
            # because last line may be incomplete
            last_line = pull_tail.splitlines()[-2].split("\t")
            try:
                latest_time = float(last_line[0])
            except IndexError:
                pass
            try:
                latest_distance = float(last_line[3])
            except IndexError:
                pass

    return LogCheckResult(
        found_error=found_error,
        error_type=error_type,
        error_msg=error_msg,
        latest_time=latest_time,
        latest_distance=latest_distance,
    )


class PullDirection(enum.Enum):
    Closer: auto
    Further: auto


def slurm_dispatch(
    cwd: str,
    job: str,
    name: str | None = None,
    force: bool = False,
    append: bool = False,
    clear_restart_files: bool = False,
    max_time: float | None = None,
    target_distance: float | None = None,
    pull_direction: None | PullDirection = None,
):
    """
    Dispatch a job to the slurm cluster

    Parameters
    ----------

    cwd
        The working directory of the job

    job
        The name of the job

    name
        The name of the job in the slurm queue

    force
        If True, the job will be dispatched even if the output file already exists

    append
        If True, the job will be dispatched even if the output file already exists
        and the log file contains no errors. The job will append to the existing
        output file.

    clear_restart_files
        If True, the job will remove all restart files before dispatching
        (step*.pdb, job-RESTART.wfn, job-slurm.out))

    max_time
        If not None, the job will be dispatched only if the latest time in the
        pullx file is less than max_time.
        Ues with append=True to continue a job that was stopped before completion.
        Recommended value is the target time of the mdp (in ps) minus one dt.
    """
    running_or_waiting = get_running_jobs()["job"]
    if name is None:
        name = f"{cwd}/{job}"
    if name in running_or_waiting:
        logger.info(f"Already waiting or running {name}. Not dispatching.")
        return
    dir = Path(cwd).resolve()
    trr = f"{job}.trr"
    target_finished = f"slurm-finished-{job}.info"
    tpr = f"{job}.tpr"

    with pushd(cwd):
        if not Path(tpr).exists():
            logger.info(f"{tpr} doesn't exist. Not dispatching.")
            return
        if Path(trr).exists() and not force and not append:
            logger.info(f"{trr} already exists. Not dispatching.")
            return
        if Path(target_finished).exists() and not append:
            logger.info(f"{target_finished} already exists. Not dispatching.")
            return

        if clear_restart_files:
            fs = [f"{job}-slurm.out", f"{job}-RESTART.wfn"]
            fs += glob("step*.pdb")
            for f in fs:
                p = Path(f)
                if p.exists():
                    os.remove(p)

        if append and Path(trr).exists():
            logcheck = check_logfiles_for_errors(job)
            error_msg = logcheck.error_msg
            found_error = logcheck.found_error
            error_type = logcheck.error_type
            latest_time = logcheck.latest_time
            latest_distance = logcheck.latest_distance
            if found_error:
                # NOTE:
                # When starting in append mode, `Pull reference` errors in the log
                # are good. It means the simulation is actually finished
                # and is already trying to pull further than the target distance.
                logger.info(f"Found previous error in {job}. Not dispatching.")
                logger.info(f"Error type: {error_type}")
                logger.info(f"Error msg: {error_msg}")
                return
            if latest_time == 0.0 and not force:
                logger.info(f"Latest time is 0.0. Not dispatching {job}.")
                return
            if max_time is not None and latest_time >= max_time:
                logger.info(
                    f"Latest time {latest_time} is >= than max_time {max_time}. Not dispatching {job}."
                )
                return
            if (
                pull_direction is not None
                and target_distance is not None
                and latest_distance is not None
            ):
                if (
                    pull_direction == PullDirection.Closer
                    and latest_distance <= target_distance
                ):
                    logger.info(
                        f"Latest distance {latest_distance} is <= than target_distance {target_distance}. Not dispatching."
                    )
                    return
                if (
                    pull_direction == PullDirection.Further
                    and latest_distance >= target_distance
                ):
                    logger.info(
                        f"Latest distance {latest_distance} is >= than target_distance {target_distance}. Not dispatching."
                    )
                    return
            logger.info(f"Appending to trajectory at time {latest_time}.")

        sh(f"ssh {CLUSTER} 'cd {dir} && sbatch -J {name} jobscript.sh {job}'")
        logger.info(f"Dispatched {job} in {cwd}.")


def show_job_ids(ds: list[dict]):
    running = []
    for d in ds:
        for k, v in d.items():
            rs = v.get("running")
            if not rs:
                continue
            for r in rs:
                running.append(read(f"{k}/{r}")[0])
    return running


def show_job_finished(ds: list[dict]):
    finished = []
    for d in ds:
        for k, v in d.items():
            rs = v.get("running")
            ws = v.get("waiting")
            if not rs and not ws:
                finished.append(k)
    return finished


def local_dispatch_systems(env, job):
    for system in ["single", "triple"]:
        cwd = env[f"{system}_dir"]
        with pushd(cwd):
            sh(f"./local-job.sh {job}")


def slurm_dispatch_systems(
    env: dict,
    job: str,
    force: bool = False,
    append: bool = False,
    clear_restart_files: bool = False,
    max_time: float | None = None,
):
    for system in ["single", "triple"]:
        system_dir = env[f"{system}_dir"]
        system_env = read_env(system_dir)
        assert system_env
        frames = system_env["frames"]
        for frame in frames:
            frame_dir = f"{system_dir}/frame-{frame}"
            slurm_dispatch(
                cwd=frame_dir,
                job=job,
                force=force,
                append=append,
                clear_restart_files=clear_restart_files,
                max_time=max_time,
            )


def delete_qm_dir(env):
    for dir in ["single", "triple"]:
        cwd = env[f"{dir}_dir"]
        sh(f"rm -rf {cwd}")

def is_float(str):
    try:
        float(str)
        return True
    except ValueError:
        return False

def read_xvg(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """ Read an xvg file and return a pandas DataFrame.
    The xvg file is expected to have a header with column names, which are used to
    infer the columns of the DataFrame if `columns` is None.

    Pass an empty list to `columns` to return numbered columns.
    """
    ls = []
    with open(path, "r") as f:
        if columns is None:
            # extract column names from the file
            columns = []
            l = f.readline()
            while l.startswith("#") or l.startswith("@"):
                # x column
                if "xaxis" in l:
                    cname = l.split('"')[1]
                    columns.append(cname)
                # y columns
                if l.startswith("@ s"):
                    cname = l.split('"')[1]
                    columns.append(cname)

                l = f.readline()

        for i,l in enumerate(f):
            if l.startswith("@") or l.startswith("#"):
                continue
            l = l.split()
            # all should be floats
            if len(l) == 0:
                continue
            if not all(is_float(x) for x in l):
                logger.warning(
                    f"Line {i} in {path} has non-float values: {l}. Ignoring line."
                )
                continue
            # ignore incomplete lines
            # usually the last line

            if len(l) != len(columns) and len(columns) != 0:
                logger.warning(
                    f"Line {i} in {path} has {len(l)} columns, expected {len(columns)}. Ignoring line:\n{l}"
                )
                continue

            ls.append(l)

    if len(columns) == 0:
        df = pd.DataFrame(ls, dtype=float)
    else:
        try:
            df = pd.DataFrame(ls, columns=pd.Series(columns), dtype=float)
        except ValueError as _:
            df = pd.DataFrame()

    return df


def activate_mpl(remote=True):
    if remote:
        mpl.rcParams["webagg.open_in_browser"] = False
        mpl.rcParams["webagg.port"] = 9999
        mpl.use("webagg")
    else:
        # export __GLX_VENDOR_LIBRARY_NAME=mesa
        mpl.use("gtk4agg")
        plt.ion()


def plot_umbrella_time_selection(ts, xs):
    plt.clf()
    plt.vlines(ts, ymin=0, ymax=0.5, colors="black", linestyles="dashed")
    sns.lineplot(xs, x="t", y="d")
    plt.show()


def get_rotation_restraints(assets, system):
    """
    Get the rotation restraints section for the mdp file
    depending on if the systme is a triple helix
    or one of the single chains.
    """
    if system == "triple":
        partial = f"{assets}/rotation-restraints.partial.mdp"
        with open(partial, "r") as f:
            return "".join(f.readlines())
    else:
        return ""


def clear_job(env, job, frames: Optional[list[str]] = None, rotref=False):
    for system in ["single", "triple"]:
        system_dir = env[f"{system}_dir"]
        system_env = read_env(system_dir)
        assert system_env
        if frames is None:
            frames = system_env["frames"]
        assert frames
        for frame in frames:
            frame_dir = f"{system_dir}/frame-{frame}"
            with pushd(frame_dir):
                jobfiles = glob(f"{job}.*") + glob(f"{job}_*") + glob(f"{job}-*")
                if rotref:
                    jobfiles += glob("rotref.*.trr")
                for f in jobfiles:
                    os.remove(f)


def archive_job(envs, job, name=None):
    if name is None:
        name = f"{pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    for env in envs:
        for system in ["single", "triple"]:
            system_dir = env[f"{system}_dir"]
            system_env = read_env(system_dir)
            assert system_env
            frames = system_env["frames"]
            assert frames
            for frame in frames:
                frame_dir = f"{system_dir}/frame-{frame}"
                archive_path = Path(f"archive/{name}/{frame_dir}/")
                job_dir = Path(frame_dir)
                jobfiles = (
                    glob(f"{job_dir/job}.*")
                    + glob(f"{job_dir/job}_*")
                    + glob(f"{job_dir/job}-*")
                    + glob(f"{job_dir}/env.json")
                )
                if len(jobfiles) <= 1:
                    continue
                archive_path.mkdir(parents=True, exist_ok=True)
                system_archive_path = Path(f"archive/{name}/{system_dir}/")
                write_env(system_env, system_archive_path / "env.json")
                for f in jobfiles:
                    logger.info(f"Archiving {f} to {archive_path}")
                    shutil.copy(f, archive_path)


def safely(fun: Callable[..., T], *args, **kwargs) -> T | None:
    """
    Call function `fun` with args and kwargs and ignore `Exception`s
    """
    try:
        return fun(*args, **kwargs)
    except Exception as e:
        logger.error(e)
        return None


def unsafe(fun: Callable[..., T | None], *args, **kwargs) -> T:
    """
    Call function `fun` with args and kwargs and panic if it returns None.
    otherwise return the result.
    """
    result = fun(*args, **kwargs)
    assert result is not None, f"{fun} returned None. Stopping."
    return result


def itersum(xs):
    return reduce(lambda x, _: x + 1, deepcopy(xs), 0)


def find_latest_t(job: str) -> tuple[int, float] | tuple[None, None]:
    """
    Find latest logged (frame, time) of a job.
    """
    logtail = shget(f"tail -n 20 {job}.log").splitlines()
    flag = False
    frame = None
    time = None
    for l in logtail:
        if flag:
            frame, time = l.split()
            break
        if "Time" in l:
            flag = True

    if frame is not None and time is not None:
        return int(frame), float(time)
    else:
        return None, None


def get_running_jobs() -> dict[str, list[str]]:
    """Return a list of running or waiting slurm jobs"""
    info = shget(f"ssh {CLUSTER} 'squeue --me --format=\"%.100j|%.R|%.10i\"'")
    if info is None:
        info = ""

    result = {"job": [], "status": [], "id": []}
    for job, status, id in [l.split("|") for l in info.split("\n")[1:-1]]:
        result["job"].append(job.strip())
        result["status"].append(status.strip())
        result["id"].append(id.strip())
    return result


def cancel_jobs(ids: list[str]):
    if len(ids) > 0:
        sh(f"ssh {CLUSTER} 'scancel {' '.join(ids)}'")


def fix_env(path: str | Path) -> dict:
    env = read_env(path)
    assert env is not None
    env["main_dir"] = "run_3"
    env["force_dir"] = env["force_dir"].replace("run_2", "run_3")
    env["peptide_bond_dir"] = env["peptide_bond_dir"].replace("run_2", "run_3")
    env["triple_dir"] = env["triple_dir"].replace("run_2", "run_3")
    env["single_dir"] = env["single_dir"].replace("run_2", "run_3")
    try:
        env["frame_dir"] = env["frame_dir"].replace("run_2", "run_3")
        env["dir"] = env["dir"].replace("run_2", "run_3")
    except:
        pass
    write_env(env, path)
    return env


def fix_envs():
    fs = glob("run_3//**/env.json", recursive=True)
    for f in fs:
        fix_env(f)


def match_global_env(envs: list[dict], cwd: str) -> dict:
    d = "/".join(cwd.split("/")[0:3])
    for env in envs:
        if env["peptide_bond_dir"] == d:
            return env

    raise ValueError(f"Couldn't find env for {cwd}")
