"""
Hamilton pipeline for hydrolysis analysis - Python translation of _targets.R
"""

import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pandas as pd
from hamilton.function_modifiers.metadata import cache

import src.steps as st
from src.analysis_utils import (
    ix_to_resname,
    read_distances,
    read_timestamp,
    read_us_hist,
    read_us_prof,
    analyse_wethyd_us_frame,
    analyse_break_distances_frame,
    analyse_break_protons_frame,
    analyse_us_comb,
)
from src.settings import N_FRAMES
from src.utils import read_xvg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("analysis.log"), logging.StreamHandler()],
)

logger = logging.getLogger("analysis")

# TODO: skipping new set until we have results for them
NEW_IXS_C = [329, 1451, 866, 214, 228]

#### Inputs and Timestampts ####

@cache(behavior="recompute")
def envs() -> list[dict[str, Any]]:
    """Read all environment files"""
    return st.setup_envs()


@cache(behavior="recompute", format="parquet")
def wetbreak_starts() -> pd.DataFrame:
    """Load wetbreak starts data"""
    path = "data/results/wetbreak_starts.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame()


@cache(format="parquet")
def wethyd_choices() -> pd.DataFrame:
    """Load and process wethyd choices data"""
    path = "data/results/wethyd_choices.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["resname_c"] = df["ix_c"].apply(ix_to_resname)
        df["resname_n"] = df["ix_n"].apply(ix_to_resname)
        df["frame_index"] = df.groupby(["system", "ix_c"])["frame"].transform(
            lambda x: pd.Categorical(x).codes + 1
        )
        df.query("ix_c not in @NEW_IXS_C", inplace=True)
        return df
    else:
        return pd.DataFrame()

@cache(behavior="recompute", format="parquet")
def wetbreak_choices() -> pd.DataFrame:
    """Load and process wetbreak choices data"""
    path = "data/results/wetbreak_starts.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["resname_c"] = df["ix_c"].apply(ix_to_resname)
        df["resname_n"] = df["ix_n"].apply(ix_to_resname)
        df["frame_index"] = df.groupby(["system", "ix_c"])["frame"].transform(
            lambda x: pd.Categorical(x).codes + 1
        )
        df.query("ix_c not in @NEW_IXS_C", inplace=True)
        return df
    else:
        return pd.DataFrame()

@cache(behavior="recompute")
def ts_hyd_energies() -> str:
    name = "hyd-energies"
    timestamp = read_timestamp(name)
    if timestamp is None:
        logger.warning(f"No timestamp found for {name}")
        return ""
    return timestamp

@cache(behavior="recompute")
def ts_hyd_us() -> str:
    name = "hyd-us"
    timestamp = read_timestamp(name)
    if timestamp is None:
        logger.warning(f"No timestamp found for {name}")
        return ""
    return timestamp

@cache(behavior="recompute")
def ts_hyd_us_comb() -> str:
    name = "hyd-us-comb"
    timestamp = read_timestamp(name)
    if timestamp is None:
        logger.warning(f"No timestamp found for {name}")
        return ""
    return timestamp

@cache(behavior="recompute")
def ts_break_us() -> str:
    name = "break-us"
    timestamp = read_timestamp(name)
    if timestamp is None:
        logger.warning(f"No timestamp found for {name}")
        return ""
    return timestamp

@cache(behavior="recompute")
def ts_break_us_comb() -> str:
    name = "break-us-comb"
    timestamp = read_timestamp(name)
    if timestamp is None:
        logger.warning(f"No timestamp found for {name}")
        return ""
    return timestamp

@cache(behavior="recompute")
def ts_break_distances() -> str:
    name = "break-distances"
    timestamp = read_timestamp(name)
    if timestamp is None:
        logger.warning(f"No timestamp found for {name}")
        return ""
    return timestamp

@cache(behavior="recompute")
def ts_break_protons() -> str:
    name = "break-protons"
    timestamp = read_timestamp(name)
    if timestamp is None:
        logger.warning(f"No timestamp found for {name}")
        return ""
    return timestamp

@cache(format="parquet")
def parameters(envs: list[dict[str, Any]]) -> pd.DataFrame:
    """Generate parameters DataFrame"""
    if not envs:
        return pd.DataFrame()

    df = pd.DataFrame(envs)

    df["run"] = df["main_dir"]
    df["run_nr"] = df["run"].str.extract(r"(\d+)").astype(int)
    df["force"] = df["single_external_force"]
    df["ix_c"] = df["triple_ix_c_carbonyl"]
    df["ix_n"] = df["triple_ix_n_peptide"]
    df["resname_c"] = df["ix_c"].apply(ix_to_resname)
    df["resname_n"] = df["ix_n"].apply(ix_to_resname)

    df = df[df["run_nr"] == 6]

    systems = ["single", "triple"]
    frames = list(range(100 - N_FRAMES + 1, 101))
    jobs = ["wethyd"]

    expanded_rows = []
    for _, row in df.iterrows():
        for system in systems:
            for frame in frames:
                for job in jobs:
                    new_row = row.copy()
                    new_row["system"] = system
                    new_row["frame"] = frame
                    new_row["job"] = job
                    cwd = f"{row['run']}/f-{row['single_external_force']}/ixs-{row['triple_ix_c_carbonyl']}-{row['triple_ix_n_peptide']}/{system}/frame-{frame}"
                    new_row["cwd"] = cwd
                    expanded_rows.append(new_row)

    return pd.DataFrame(expanded_rows)

@cache(behavior="recompute")
def frame_cwds(wethyd_choices: pd.DataFrame) -> list[str]:
    cwds = []
    for cwd, ix_c in zip(wethyd_choices["cwd"], wethyd_choices["ix_c"]):
        cwds.append(cwd)
    return cwds

#### Results ####

@cache(format="parquet")
def example_equilibration() -> pd.DataFrame:
    path = f"./run_6/f-603/triple-eq_pullx.xvg"
    if os.path.exists(path):
        df = read_xvg(path, ["time", "x1", "x2"])
        df["d"] = df["x2"] - df["x1"]
        df = df.drop(columns=["x1", "x2"])
        return df
    else:
        return pd.DataFrame()


def _f(cwd: str) -> pd.DataFrame:
    try:
        return analyse_wethyd_us_frame(cwd)
    except Exception as e:
        logger.error(f"Error processing {cwd}: {e}")
        return pd.DataFrame()

@cache(format="parquet")
def wethyd_us_energies(frame_cwds: list[str], ts_hyd_energies: str) -> pd.DataFrame:
    _ = ts_hyd_energies
    all_frames: list[pd.DataFrame] = []
    with Pool() as p:
        all_frames = p.map(_f, frame_cwds)

    combined = pd.concat(all_frames, ignore_index=True)
    return combined

def _f_break(cwd: str) -> pd.DataFrame:
    try:
        return analyse_break_distances_frame(cwd)
    except Exception as e:
        logger.error(f"Error processing {cwd}: {e}")
        return pd.DataFrame()

@cache(format="parquet")
def break_us_distances(frame_cwds: list[str], ts_break_distances: str) -> pd.DataFrame:
    _ = ts_break_distances
    all_frames: list[pd.DataFrame] = []
    with Pool() as p:
        all_frames = p.map(_f_break, frame_cwds)

    combined = pd.concat(all_frames, ignore_index=True)
    return combined

def _f_break_protons(cwd: str) -> pd.DataFrame:
    try:
        return analyse_break_protons_frame(cwd)
    except Exception as e:
        logger.error(f"Error processing {cwd}: {e}")
        return pd.DataFrame()

@cache(format="parquet")
def break_us_protons(frame_cwds: list[str], ts_break_protons: str) -> pd.DataFrame:
    _ = ts_break_protons
    all_frames: list[pd.DataFrame] = []
    with Pool() as p:
        all_frames = p.map(_f_break_protons, frame_cwds)

    combined = pd.concat(all_frames, ignore_index=True)
    return combined


@cache(format="parquet")
def hyd_pull(parameters: pd.DataFrame) -> pd.DataFrame:
    job = "wethyd"

    pull_data = pd.DataFrame()
    for _, row in parameters.iterrows():
        cwd = row['cwd']
        path_pullx = Path(f"{cwd}/{job}_pullx.xvg")
        path_pullf = Path(f"{cwd}/{job}_pullf.xvg")
        if not path_pullx.exists() or not path_pullf.exists():
            continue

        pullx = read_xvg(path=path_pullx, columns=["time", "x1", "x2", "x3", "x4"])
        pullf = read_xvg(path=path_pullf, columns=["time", "f1", "f2", "f3", "f4"])
        if not pullx.empty and not pullf.empty:
            pullx["time"] = pullx["time"].astype(float)
            pullx.set_index("time", inplace=True)
            pullf["time"] = pullf["time"].astype(float)
            pullf.set_index("time", inplace=True)
            pullx = pullx.rename(columns={"x3": "x"}).filter(items=["time", "x"])
            pullf = pullf.rename(columns={"f3": "f"}).filter(items=["time", "f"])
            merged = pullx.join(pullf, how="inner").reset_index()
            merged["cwd"] = cwd
            pull_data = pd.concat([pull_data, merged], ignore_index=True)

    if pull_data.empty:
        return pd.DataFrame()

    return parameters.merge(pull_data, on="cwd", how="right")


@cache(format="parquet")
def wethyd_us_hist(
    wethyd_choices: pd.DataFrame, ts_hyd_us: str
) ->pd.DataFrame:
    """Process umbrella sampling data"""
    _ = ts_hyd_us
    hists = []

    df = wethyd_choices.copy()
    df["path_hist"] = df["cwd"] + "/wethyd-hist.xvg"

    for _, row in df.iterrows():
        path: Path = Path(row["path_hist"])
        if not path.exists():
            hist_data = pd.DataFrame()
        else:
            hist_data = read_us_hist(path=path)

        for col in row.index:
            if col not in ["path_hist", "path_prof"]:
                hist_data[col] = row[col]

        hists.append(hist_data)

    if len(hists) > 0:
        hist = pd.concat(hists, ignore_index=True)
        return hist
    else:
        return pd.DataFrame()

@cache(format="parquet")
def wethyd_us_prof(
    wethyd_choices: pd.DataFrame, ts_hyd_us: str
) -> pd.DataFrame:
    """Process umbrella sampling data"""
    _ = ts_hyd_us
    profs = []

    df = wethyd_choices.copy()
    df["path_prof"] = df["cwd"] + "/wethyd-bsResult.xvg"

    for _, row in df.iterrows():
        path: Path = Path(row["path_prof"])
        if not path.exists():
            prof_data = pd.DataFrame()
        else:
            prof_data = read_us_prof(path=path)

        for col in row.index:
            if col not in ["path_hist", "path_prof"]:
                prof_data[col] = row[col]

        profs.append(prof_data)

    if len(profs) > 0:
        prof = pd.concat(profs, ignore_index=True)
        return prof
    else:
        return pd.DataFrame()



def wethyd_us_comb(
    wethyd_choices: pd.DataFrame, ts_hyd_us_comb: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process combined umbrella sampling data"""
    _ = ts_hyd_us_comb
    if wethyd_choices.empty:
        return pd.DataFrame(), pd.DataFrame()

    choices = wethyd_choices.copy()
    return analyse_us_comb(parent_job="wethyd", choices=choices)

def wetbreak_us_comb(
    wetbreak_choices: pd.DataFrame, ts_break_us_comb: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process combined umbrella sampling data"""
    _ = ts_break_us_comb
    if wetbreak_choices.empty:
        return pd.DataFrame(), pd.DataFrame()

    choices = wetbreak_choices.copy()
    return analyse_us_comb(parent_job="wetbreak", choices=choices)


def wetbreak_us(
    wetbreak_choices: pd.DataFrame, ts_break_us: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process umbrella sampling data"""
    _ = ts_break_us
    hists = []
    profs = []

    df = wetbreak_choices.copy()
    df["path_hist"] = df["cwd"] + "/wetbreak-hist.xvg"
    df["path_prof"] = df["cwd"] + "/wetbreak-bsResult.xvg"

    for _, row in df.iterrows():
        p = row["path_hist"]
        path: Path = Path(p)
        if not path.exists():
            logger.warning(f"Umbrella sampling hist file {path} does not exist.")
            hist_data = pd.DataFrame()
        else:
            hist_data = read_us_hist(path=path)

        p = row["path_prof"]
        path: Path = Path(p)
        if not path.exists():
            logger.warning(f"Umbrella sampling hist file {path} does not exist.")
            prof_data = pd.DataFrame()
        else:
            prof_data = read_us_prof(path=path)

        for col in row.index:
            if col not in ["path_hist", "path_prof"]:
                prof_data[col] = row[col]
                hist_data[col] = row[col]

        profs.append(prof_data)
        hists.append(hist_data)

    if len(hists) > 0 and len(profs) > 0:
        hist = pd.concat(hists, ignore_index=True)
        prof = pd.concat(profs, ignore_index=True)
        return hist, prof
    else:
        return pd.DataFrame(), pd.DataFrame()

