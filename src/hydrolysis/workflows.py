import pandas as pd
from tqdm import tqdm

import hydrolysis.steps as st
from hydrolysis.settings import HYD_ATTACK_CUTOFF


def check_wethyd_outcomes(envs):
    outcomes = []
    for env in tqdm(envs, desc="envs"):
        outcomes += st.check_wethyd_outcomes(env)

    df = pd.DataFrame(outcomes)
    df["msg"] = df["msg"].str.replace("\n", "\\n")
    df["cwd"] = df.apply(
        lambda l: f"{l.run}/f-{l.external_force}/ixs-{l.ix_c}-{l.ix_n}/{l.system}/frame-{l.frame}",
        axis=1,
    )
    df.to_csv("./data/results/wethyd_outcomes.csv", lineterminator="\n", index=False)

    choices = df[df.d < HYD_ATTACK_CUTOFF]
    choices.to_csv(
        "./data/results/wethyd_choices.csv", lineterminator="\n", index=False
    )
    wetbreak_starts = (
        choices.groupby(["job", "run", "external_force", "ix_c", "ix_n", "system"])
        .nth(0)
        .reset_index()
    )
    wetbreak_starts.to_csv(
        "./data/results/wetbreak_starts.csv", lineterminator="\n", index=False
    )

    return df


def check_wethyd_us_outcomes(envs):
    outcomes = []
    for env in tqdm(envs, desc="envs"):
        outcomes += st.check_wethyd_us_outcomes(env)

    df = pd.DataFrame(outcomes)
    df["msg"] = df["msg"].str.replace("\n", "\\n")
    df.to_csv("./data/results/wethyd_us_outcomes.csv", lineterminator="\n", index=False)

    return df


def check_wethyd_warmup_outcomes(envs):
    outcomes = []
    for env in tqdm(envs, desc="envs"):
        outcomes += st.check_wethyd_warmup_outcomes(env)

    df = pd.DataFrame(outcomes)
    df["msg"] = df["msg"].str.replace("\n", "\\n")
    df["cwd"] = df.apply(
        lambda l: f"{l.run}/f-{l.external_force}/ixs-{l.ix_c}-{l.ix_n}/{l.system}/frame-{l.frame}",
        axis=1,
    )
    df.to_csv(
        "./data/results/wethyd_warmup_outcomes.csv", lineterminator="\n", index=False
    )
    return df
