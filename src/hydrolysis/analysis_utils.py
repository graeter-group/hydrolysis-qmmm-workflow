import pandas as pd
import os
from pathlib import Path

from hydrolysis.utils import read_xvg, pushd, read_env
from hydrolysis.constants import N_QW, HBOND
from datetime import datetime
from numpy import Inf
import logging

logger = logging.getLogger(__name__)

ix_cols = ["ix_c", "ix_n", "system", "conf_d", "conf_t", "frame"]

def write_timestamp(job: str) -> None:
    """Write the current timestamp to a file"""
    path = f".{job}-timestamp.info"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w") as f:
        f.write(timestamp)
        logger.info(f"Timestamp {timestamp} written to {path}")

def read_timestamp(job: str) -> str | None:
    """Read the timestamp from a file"""
    path = f".{job}-timestamp.info"
    if not Path(path).exists():
        return None
    with open(path, "r") as f:
        timestamp = f.read().strip()
    return timestamp



def read(p: str | Path) -> list[str]:
    with open(p, "r") as f:
        lines = f.readlines()
    return lines


# Helper functions translated from R/global.r
def get_env_paths(env_dir: str = "envs") -> list[str]:
    """Get paths to environment JSON files"""
    env_path = Path(env_dir)
    return [str(p) for p in env_path.glob("*.json")]


def read_distances(cwd: str|Path, job: str) -> pd.DataFrame:
    """Read distance files and combine them"""
    try:
        path_oc = f"{cwd}/{job}-mindist-o-c.xvg"
        path_oh = f"{cwd}/{job}-mindist-o-h.xvg"
        path_cqm = f"{cwd}/{job}-mindist-c-w.xvg"

        oc_data = read_xvg(path_oc, ["time", "O,C"])
        oh_data = read_xvg(path_oh, ["time", "O,H"])
        cqm_cols = ["time"] + [f"C,QW_{i}" for i in range(1, N_QW + 2)]
        cqm_data = read_xvg(path_cqm, cqm_cols)

        # Merge dataframes
        result = oc_data.merge(oh_data, on="time", how="left")
        result = result.merge(cqm_data, on="time", how="left")

        # Pivot to long format
        result = result.melt(id_vars=["time"], var_name="pair", value_name="dist")
        return result
    except Exception:
        return pd.DataFrame()


def read_us_hist(path: str|Path) -> pd.DataFrame:
    """Read umbrella sampling histogram file"""
    if not Path(path).exists():
        logger.warning(f"Umbrella sampling hist file {path} does not exist.")
        return pd.DataFrame()
    data = read_xvg(path=path, columns=[])
    if data.empty:
        logger.warning(f"Umbrella sampling hist file {path} is empty.")
        return data

    x_col: int = data.columns[0] # type: ignore
    result = data.melt(id_vars=[x_col], var_name="window", value_name="count")
    result.rename(columns={x_col: "x"}, inplace=True)
    result["window"] = pd.to_numeric(result["window"], errors="coerce")
    return result


def read_us_prof(path: str|Path) -> pd.DataFrame:
    """Read umbrella sampling profile file"""
    if not Path(path).exists():
        logger.warning(f"Umbrella sampling profile file {path} does not exist.")
        return pd.DataFrame()
    data = read_xvg(path, columns=[])
    if data.empty:
        logger.warning(f"Umbrella sampling profile file {path} is empty.")
        return data
    if data.shape[1] == 3:
        data.columns = ["x", "E", "dE"]
    elif data.shape[1] == 2:
        data.columns = ["x", "E"]
    else:
        logger.warning(f"Umbrella sampling profile file {path} has unexpected number of columns: {data.shape[1]}")
        return pd.DataFrame()

    return data


def ix_to_resname(ix: int) -> str:
    """Convert index to residue name"""
    resnames = {
        178: "Gly842",
        180: "Leu843",
        276: "Ser850",
        278: "Gly851",
        1473: "Gly857",
        1475: "Ala858",
        802: "Ala845",
        804: "Gly846",
        745: "Gly840",
        747: "Ala841",
        1466: "Hyp856",
        1468: "Gly857",
        881: "Hyp851",
        883: "Gly852",
        918: "Gly855",
        920: "Pro856",
        329: "Gly854",
        331: "Ser855",
        1451: "Ser855",
        1453: "Hyp856",
        866: "Asn850",
        868: "Hyp851",
        214: "Gly845",
        216: "Pro846",
        228: "Pro846",
        230: "Hyp847",
    }
    return resnames.get(ix, "unknown")

def analyse_wethyd_us_frame(cwd: str) -> pd.DataFrame:
    frame = pd.DataFrame()

    meta = cwd.split("/")
    run_dir, force, ixs, system, frame_dir = meta
    external_force = int(force.removeprefix("f-"))
    ix_c, ix_n = map(int, ixs.split("-")[1:])
    frame_nr = int(frame_dir.split("-")[1])
    run_nr = int(run_dir.split("_")[-1])
    logger.info(
        f"Processing run {run_nr} frame {frame_nr} with force {external_force}, ix_c {ix_c}, ix_n {ix_n}"
    )
    if not os.path.exists(cwd):
        logging.warning(f"Directory {cwd} does not exist.")
        return frame

    with pushd(cwd):
        env = read_env()
        if not env:
            return frame
        ts = env["wethyd-config-times"]
        ds = env["wethyd-config-ds"]
        if not ts or not ds:
            return frame
        ts_ds = list(zip(ts, ds))
        ts_ds.sort(key=lambda x: float(x[1]))
        path_sel_o = "protdist-o.sel"
        path_sel_h = "protdist-h.sel"
        if not Path(path_sel_o).exists() or not Path(path_sel_h).exists():
            logger.warning(
                f"Selection files {path_sel_o} or {path_sel_h} do not exist in {cwd}."
            )
            return frame

        # includes the N of the peptide bond as a proton acceptor
        ids_o = [
            int(x) for x in
            read(path_sel_o)[0]
            .removeprefix("atomnr")
            .removesuffix(";")
            .strip()
            .split()
        ]
        ids_o.sort()
        ids_h = [
            int(x) for x in
             read(path_sel_h)[0]
            .removeprefix("atomnr")
            .removesuffix(";")
            .strip()
            .split()
        ]
        ids_h.sort()
        # gromacs prints the pair distances with the ids
        # in the selection file ordered because
        # it is one selection with grouping none
        # See <https://manual.gromacs.org/documentation/current/onlinehelp/gmx-pairdist.html>
        # The columns contain distances like this: r1-s1, r2-s1, …, r1-s2, r2-s2, …,
        # where rn is the n’th group in -ref and sn is the n’th group in the other selection.
        colnames = ["time"]
        for h in ids_h:
            for o in ids_o:
                colnames.append(f"{o}_{h}")

        ix_c_carbonyl = int(env["ix_c_carbonyl"])
        ix_o_carbonyl = int(env["ix_o_carbonyl"])
        ix_o_oh = int(env["ix_o_oh"])
        ix_n_peptide = int(env["ix_n_peptide"])

        id_c_carbonyl = ix_c_carbonyl + 1
        id_o_carbonyl = ix_o_carbonyl + 1
        id_o_oh = ix_o_oh + 1
        id_n_peptide = ix_n_peptide + 1

        for conf_t, conf_d in ts_ds:
            job = f"wethyd-conf-{conf_t}"

            path_pullx = f"{job}_pullx.xvg"
            path_pullf = f"{job}_pullf.xvg"
            protdist_path = f"{job}-protdist.xvg"
            tidist_path = f"{job}-tidist.xvg"
            tistabilty_path = f"{job}-ti-stability.xvg"
            energy_path = f"{job}-energy.xvg"

            paths_exist = [Path(p).exists() for p in [
                path_pullx, path_pullf, protdist_path, tidist_path, energy_path
            ]]
            if not all(paths_exist):
                logger.warning(
                    f"One or more required files do not exist for conf_t {conf_t} in {cwd}: "
                    f"{', '.join([p for p, exists in zip([path_pullx, path_pullf, protdist_path, tidist_path, energy_path], paths_exist) if not exists])}"
                )
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

            raw_protdist = read_xvg(protdist_path, colnames)
            raw_protdist["time"] = raw_protdist["time"].astype(float)
            raw_protdist.set_index("time", inplace=True)

            # for each time point, assign each h to the closest o
            ls = []
            ls_hbond = []
            for time in raw_protdist.index:
                n_h_per_o = {id: 0 for id in ids_o}
                for id_h in ids_h:
                    closest_o = None
                    closest_dist = Inf
                    for id_o in ids_o:
                        col = f"{id_o}_{id_h}"
                        d: float = raw_protdist.at[time, col]
                        if d < closest_dist:
                            closest_dist = d
                            closest_o = id_o
                    if closest_o is not None:
                        n_h_per_o[closest_o] += 1
                    else:
                        raise ValueError(
                            f"No closest O found for H {id_h} at time {conf_t}"
                        )
                # track who has the proton covering a negative charge
                # O of OH- needs 2 H in order to be neutral
                # C of carbonyl counts as having a proton if it has >=1 H
                # the 14 QM waters would be neutral if they have 28 H
                l = {'time': time,
                     'n_h_oh': n_h_per_o[id_o_oh],
                     'n_h_oc': n_h_per_o[id_o_carbonyl],
                     'n_h_np': n_h_per_o[id_n_peptide],
                     'n_h_qw': sum(n_h_per_o.values()) - n_h_per_o[id_o_oh] - n_h_per_o[id_o_carbonyl] - n_h_per_o[id_n_peptide]
                     }
                ls.append(l)

                # also track how many potential H-bonds each O has
                # from the same raw distance information
                n_hb_per_o = {id: 0 for id in ids_o}
                for id_o in ids_o:
                    for id_h in ids_h:
                        col = f"{id_o}_{id_h}"
                        d: float = raw_protdist.at[time, col]
                        if d <= HBOND:
                            n_hb_per_o[id_o] += 1

                    # subtract the protons already assigned to this O
                    # as being bound to it
                    # TODO: why do we get negative numbers here?

                    # n_hb_per_o[id_o] -= n_h_per_o[id_o]
                    # if n_hb_per_o[id_o] < 0:
                    #     logger.warning(
                    #         f"Negative number of H-bonds for O {id_o} at time {time} in conf_t {conf_t}. Setting to 0."
                    #     )
                    #     n_hb_per_o[id_o] = 0

                # track just the total number of hbonds
                l_hb = {'time': time,
                        'n_hb': sum(n_hb_per_o.values())
                        }
                ls_hbond.append(l_hb)


            protdist = pd.DataFrame(ls)
            protdist["time"] = protdist["time"].astype(float)
            protdist.set_index("time", inplace=True)

            hbonds = pd.DataFrame(ls_hbond)
            hbonds["time"] = hbonds["time"].astype(float)
            hbonds.set_index("time", inplace=True)

            tidist = read_xvg(tidist_path, ["time", "qmh", "proteinh", "solh"])
            tidist["time"] = tidist["time"].astype(float)
            tidist.set_index("time", inplace=True)

            tistability = read_xvg(tistabilty_path, ["time", "c_o_carb", "c_o_oh"])
            tistability["time"] = tistability["time"].astype(float)
            tistability.set_index("time", inplace=True)

            energy = read_xvg(energy_path)
            energy.rename({"Time (ps)": "time"}, axis=1, inplace=True)
            energy["time"] = energy["time"].astype(float)

            if not pullx.empty:
                energy = energy.join(pullx, on="time", how="left")
            if not pullf.empty:
                energy = energy.join(pullf, on="time", how="left")
            if not protdist.empty:
                energy = energy.join(protdist, on="time", how="left")
            if not hbonds.empty:
                energy = energy.join(hbonds, on="time", how="left")
            if not tidist.empty:
                energy = energy.join(tidist, on="time", how="left")
            if not tistability.empty:
                energy = energy.join(tistability, on="time", how="left")

            energy["conf_t"] = conf_t
            energy["conf_d"] = conf_d
            energy["cwd"] = cwd
            energy["run"] = run_dir
            energy["external_force"] = external_force
            energy["system"] = system
            energy["frame"] = frame_nr
            energy["ix_c"] = ix_c
            energy["ix_n"] = ix_n
            energy.reset_index(inplace=True)
            frame = pd.concat([frame, energy], ignore_index=True)

    return frame

def analyse_break_protons_frame(cwd: str) -> pd.DataFrame:
    frame = pd.DataFrame()
    parentjob = "wetbreak"

    meta = cwd.split("/")
    run_dir, force, ixs, system, frame_dir = meta
    external_force = int(force.removeprefix("f-"))
    ix_c, ix_n = map(int, ixs.split("-")[1:])
    frame_nr = int(frame_dir.split("-")[1])
    run_nr = int(run_dir.split("_")[-1])
    logger.info(
        f"Processing run {run_nr} frame {frame_nr} with force {external_force}, ix_c {ix_c}, ix_n {ix_n}"
    )
    if not os.path.exists(cwd):
        logging.warning(f"Directory {cwd} does not exist.")
        return frame

    with pushd(cwd):
        env = read_env()
        if not env:
            return frame
        ts = env[f"{parentjob}-config-times"]
        ds = env[f"{parentjob}-config-ds"]
        if not ts or not ds:
            return frame
        ts_ds = list(zip(ts, ds))
        ts_ds.sort(key=lambda x: float(x[1]))

        refpath = "protdist-break-o.sel"
        selpath = "protdist-break-h.sel"
        if not Path(refpath).exists() or not Path(selpath).exists():
            logger.warning(
                f"Selection files {refpath} or {selpath} do not exist in {cwd}."
            )
            return frame

        ids_o = [
            int(x) for x in
            read(refpath)[0]
            .removeprefix("atomnr")
            .removesuffix(";")
            .strip()
            .split()
        ]
        ids_o.sort()
        ids_h = [
            int(x) for x in
             read(selpath)[0]
            .removeprefix("atomnr")
            .removesuffix(";")
            .strip()
            .split()
        ]
        ids_h.sort()
        # gromacs prints the pair distances with the ids
        # in the selection file ordered because
        # it is one selection with grouping none
        # See <https://manual.gromacs.org/documentation/current/onlinehelp/gmx-pairdist.html>
        # The columns contain distances like this: r1-s1, r2-s1, …, r1-s2, r2-s2, …,
        # where rn is the n’th group in -ref and sn is the n’th group in the other selection.
        colnames = ["time"]
        for h in ids_h:
            for o in ids_o:
                colnames.append(f"{o}_{h}")

        ix_c_carbonyl = int(env["ix_c_carbonyl"])
        ix_o_carbonyl = int(env["ix_o_carbonyl"])
        ix_o_oh = int(env["ix_o_oh"])
        ix_n_peptide = int(env["ix_n_peptide"])

        id_o_carbonyl = ix_o_carbonyl + 1
        id_o_oh = ix_o_oh + 1
        id_n_peptide = ix_n_peptide + 1

        for conf_t, conf_d in ts_ds:
            job = f"{parentjob}-conf-{conf_t}"
            path_pullx = f"{job}_pullx.xvg"
            path_pullf = f"{job}_pullf.xvg"
            pullx = read_xvg(path=path_pullx, columns=["time", "x1", "x2", "x3", "x4"])
            pullf = read_xvg(path=path_pullf, columns=["time", "f1", "f2", "f3", "f4"])
            if not pullx.empty and not pullf.empty:
                pullx["time"] = pullx["time"].astype(float)
                pullx.set_index("time", inplace=True)
                pullf["time"] = pullf["time"].astype(float)
                pullf.set_index("time", inplace=True)
                pullx = pullx.rename(columns={"x3": "x"}).filter(items=["time", "x"])
                pullf = pullf.rename(columns={"f3": "f"}).filter(items=["time", "f"])

            protdist_path = f"{job}-protdist.xvg"
            raw_protdist = read_xvg(protdist_path, colnames)
            raw_protdist["time"] = raw_protdist["time"].astype(float)
            raw_protdist.set_index("time", inplace=True)

            # for each time point, assign each h to the closest o or N
            ls = []
            for time in raw_protdist.index:
                n_h_per_o = {id: 0 for id in ids_o}
                for id_h in ids_h:
                    closest_o = None
                    closest_dist = Inf
                    for id_o in ids_o:
                        col = f"{id_o}_{id_h}"
                        d: float = raw_protdist.at[time, col]
                        if d < closest_dist:
                            closest_dist = d
                            closest_o = id_o
                    if closest_o is not None:
                        n_h_per_o[closest_o] += 1
                    else:
                        raise ValueError(
                            f"No closest O found for H {id_h} at time {conf_t}"
                        )
                # track who has the proton covering a negative charge
                # O of OH- needs 2 H in order to be neutral
                # C of carbonyl counts as having a proton if it has >=1 H
                # the 14 QM waters would be neutral if they have 28 H
                l = {'time': time,
                     'n_h_oh': n_h_per_o[id_o_oh],
                     'n_h_oc': n_h_per_o[id_o_carbonyl],
                     'n_h_n': n_h_per_o[id_n_peptide],
                     'n_h_qw': sum(n_h_per_o.values()) - n_h_per_o[id_o_oh] - n_h_per_o[id_o_carbonyl] - n_h_per_o[id_n_peptide]
                     }
                ls.append(l)

            protdist = pd.DataFrame(ls)
            protdist["time"] = protdist["time"].astype(float)
            protdist.set_index("time", inplace=True)

            pull = pullx
            pull = pull.join(pullf, on="time", how="left")
            pull = pull.join(protdist, on="time", how="left")

            pull["conf_t"] = conf_t
            pull["conf_d"] = conf_d
            pull["cwd"] = cwd
            pull["run"] = run_dir
            pull["external_force"] = external_force
            pull["system"] = system
            pull["frame"] = frame_nr
            pull["ix_c"] = ix_c
            pull["ix_n"] = ix_n
            pull.reset_index(inplace=True)
            frame = pd.concat([frame, pull], ignore_index=True)

    return frame

def analyse_break_distances_frame(cwd: str) -> pd.DataFrame:
    refpath = f"break-ti-distances-ref.sel"
    selpath = f"break-ti-distances-sel.sel"
    parentjob = "wetbreak"

    frame = pd.DataFrame()
    meta = cwd.split("/")
    run_dir, force, ixs, system, frame_dir = meta
    external_force = int(force.removeprefix("f-"))
    ix_c, ix_n = map(int, ixs.split("-")[1:])
    frame_nr = int(frame_dir.split("-")[1])
    run_nr = int(run_dir.split("_")[-1])
    logger.info(
        f"Processing run {run_nr} frame {frame_nr} with force {external_force}, ix_c {ix_c}, ix_n {ix_n}"
    )
    if not os.path.exists(cwd):
        logging.warning(f"Directory {cwd} does not exist.")
        return frame

    with pushd(cwd):
        env = read_env()
        if not env:
            return frame
        ts = env[f"{parentjob}-config-times"]
        ds = env[f"{parentjob}-config-ds"]
        if not ts or not ds:
            return frame
        ts_ds = list(zip(ts, ds))
        ts_ds.sort(key=lambda x: float(x[1]))
        if not Path(refpath).exists() or not Path(selpath).exists():
            logger.warning(
                f"Selection files {refpath} or {selpath} do not exist in {cwd}."
            )
            return frame

        for conf_t, conf_d in ts_ds:
            job = f"{parentjob}-conf-{conf_t}"
            target = f"{job}-break-ti-distances.xvg"

            tistability = read_xvg(target, ["time", "c_o_carb", "c_o_oh", "c_n_pep"])
            tistability["time"] = tistability["time"].astype(float)
            tistability.set_index("time", inplace=True)

            tistability["conf_t"] = conf_t
            tistability["conf_d"] = conf_d
            tistability["cwd"] = cwd
            tistability["run"] = run_dir
            tistability["external_force"] = external_force
            tistability["system"] = system
            tistability["frame"] = frame_nr
            tistability["ix_c"] = ix_c
            tistability["ix_n"] = ix_n
            tistability.reset_index(inplace=True)
            frame = pd.concat([frame, tistability], ignore_index=True)

    return frame

def analyse_us_comb(parent_job: str, choices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Remove last directory component from cwd
    choices["cwd"] = choices["cwd"].str.replace(r"/[^/]+$", "", regex=True)

    # Remove specific columns and get distinct rows
    cols_to_remove = ["frame", "error", "msg", "type", "frame_nr", "frame_index", "time", "d"]
    existing_cols_to_remove = [col for col in cols_to_remove if col in choices.columns]
    choices = choices.drop(columns=existing_cols_to_remove)
    choices = choices.drop_duplicates(subset=["cwd"])

    # Add paths
    choices["path_hist"] = choices["cwd"] + f"/{parent_job}-hist.xvg"
    choices["path_prof_bs"] = choices["cwd"] + f"/{parent_job}-bsResult.xvg"
    choices["path_prof"] = choices["cwd"] + f"/{parent_job}-profile.xvg"

    hists = []
    profs = []
    for _, row in choices.iterrows():
        p = row["path_hist"]
        path: Path = Path(p)  # type: ignore
        if not path.exists():
            logger.warning(f"Umbrella sampling hist file {path} does not exist.")
            hist_data = pd.DataFrame()
        else:
            hist_data = read_us_hist(path=path)

        p = row["path_prof_bs"]
        path: Path = Path(p)  # type: ignore
        if not path.exists():
            logger.warning(f"Umbrella sampling hist file {path} does not exist.")
            prof_data = pd.DataFrame()
        else:
            prof_data = read_us_prof(path=path)

        # Add metadata
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

def apply_group_names(df: pd.DataFrame) -> pd.DataFrame:
    df["resname_c"] = df["ix_c"].apply(ix_to_resname)
    df["resname_n"] = df["ix_n"].apply(ix_to_resname)
    df['conf_d'] = df['conf_d'].astype(float)
    df["group"] = df["ix_c"].astype(str) + "-" + df["ix_n"].astype(str) + "-" + df["system"].astype(str) + "-" + df["frame"].astype(str) + "-" + df["type"].astype(str)
    df["resnames"] = df["resname_c"].astype(str) + "-" + df["resname_n"].astype(str)
    df["sys"] = df["ix_c"].astype(str) + "-" + df["ix_n"].astype(str) + "-" + df["system"].astype(str)
    df["sys2"] = df["resname_c"].astype(str) + "-" + df["resname_n"].astype(str) + "-" + df["system"].astype(str)
    df["sys_type"] = df["system"].astype(str) + "-" + df["type"].astype(str)
    return df

def melt_tiprots(tiprots: pd.DataFrame) -> pd.DataFrame:
    """Melt the tiprots dataframe to long format."""
    df = tiprots.melt(
        id_vars=ix_cols,
        value_vars=["n_h_oh", "n_h_oc", "n_h_qw", "n_h_np"],
        var_name="type",
        value_name="n",
    )
    df = df.query("type != 'n_h_qw'")
    df = apply_group_names(df)
# better name for the categories
    df['type'] = df['type'].map({'n_h_oh': 'hydroxyl O', 'n_h_oc': 'carbonyl O', 'n_h_np': 'peptide N'}) # pyright: ignore
    df['type'] = df['type'].astype("category")
    return df


