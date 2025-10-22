from pathlib import Path
import os

N_FRAMES = 10
HYD_ATTACK_CUTOFF = 0.145
ASSETS = Path(os.getcwd() + "/assets").absolute()

# Use Benedikts force: 1806 per triplehelix = 602 per single peptide = 1nN.


def get_parameters():
    """Returns a list of dictionaries with parameters for each run.
    This can then be used with `product` to get all combinations of parameters.
    """
    return [
        [
            #  run 2 to 5 share the same equilibration (manually by copying the directory before proceeding),
            # thus the same starting frames for wethyd
            # {
            #     "main_dir": "run_1",
            #     "basis_set": "DZVP-MOLOPT-GTH",
            #     "xc_functional": "PBE",
            # },
            # {
            #     "main_dir": "run_2",
            #     "basis_set": "TZVP-MOLOPT-GTH",
            #     "xc_functional": "PBE",
            # },
            # {
            #     "main_dir": "run_3",
            #     "basis_set": "TZV2P-MOLOPT-GTH",
            #     "xc_functional": "PBE",
            # },
            # {
            #     "main_dir": "run_4",
            #     "basis_set": "DZVP-MOLOPT-GTH",
            #     "xc_functional": "PBE",
            # },
            # {
            #     "main_dir": "run_5",
            #     "basis_set": "DZVP-MOLOPT-GTH",
            #     "xc_functional": "B3LYP",
            # },
            {
                "main_dir": "run_6",  # this is the one!
                "basis_set": "TZV2P-MOLOPT-GTH",
                "xc_functional": "PBE",
            },
        ],
        [
            {
                "triple_ix_c_carbonyl": "178",  # GLY842
                "triple_ix_n_peptide": "180",  # LEU843
            },
            {
                "triple_ix_c_carbonyl": "276",  # SER850
                "triple_ix_n_peptide": "278",  # GLY851
            },
            {
                "triple_ix_c_carbonyl": "1473",  # GLY857
                "triple_ix_n_peptide": "1475",  # ALA858
            },
            {
                "triple_ix_c_carbonyl": "802",  # ALA845
                "triple_ix_n_peptide": "804",  # GLY846
            },
            {
                "triple_ix_c_carbonyl": "745",  # GLY840
                "triple_ix_n_peptide": "747",  # ALA841
            },
            {
                "triple_ix_c_carbonyl": "1466",  # HYP856
                "triple_ix_n_peptide": "1468",  # GLY857
            },
            {
                "triple_ix_c_carbonyl": "881",  # HYP851
                "triple_ix_n_peptide": "883",  # GLY852
            },
            {
                "triple_ix_c_carbonyl": "918",  # GLY855
                "triple_ix_n_peptide": "920",  # PRO856
            },
            # new
            {
                "triple_ix_c_carbonyl": "329",  # GLY854
                "triple_ix_n_peptide": "331",  # SER855
            },
            {
                "triple_ix_c_carbonyl": "1451",  # SER855
                "triple_ix_n_peptide": "1453",  # HYP856
            },
            {
                "triple_ix_c_carbonyl": "866",  # ASN850
                "triple_ix_n_peptide": "868",  # HYP851
            },
            {
                "triple_ix_c_carbonyl": "214",  # GLY845
                "triple_ix_n_peptide": "216",  # PRO846
            },
            {
                "triple_ix_c_carbonyl": "228",  # PRO846
                "triple_ix_n_peptide": "230",  # HYP847
            },
        ],
        [
            {
                "single_external_force": "603",  # 603 per = 1nN per = 3 nN, 1806 for triple
                "n_steps_eq": "50000",
            },
            # {
            # 'single_external_force': '1086', # 1086 per = 1.8nN per = 5.4 nN, 3258 for triple
            # 'n_steps_eq': '50000',
            # },
            # {
            # 'single_external_force': '301', # 301 per = 0.5nN per = 1.5 nN, 903 for triple
            # 'n_steps_eq': '50000',
            # },
            # {
            # 'single_external_force': '60', # 60 per = 0.1nN per = 0.3 nN, 180 for triple
            # 'n_steps_eq': '50000',
            # },
            # {
            # 'single_external_force': '6', # 6 per = 0.01nN per = 0.03 nN, 18 for triple
            # 'n_steps_eq': '50000',
            # },
        ],
    ]


def derive_env_dirs(env, archive_path=None):
    """Derive subdirectories from the main directory and the environment variables."""
    if archive_path is not None:
        env["main_dir"] = f"{archive_path}/{env['main_dir']}"
    env["force_dir"] = env["main_dir"] + f'/f-{env["single_external_force"]}'
    env["peptide_bond_dir"] = (
        env["force_dir"]
        + f'/ixs-{env["triple_ix_c_carbonyl"]}-{env["triple_ix_n_peptide"]}'
    )
    env["triple_external_force"] = str(int(env["single_external_force"]) * 3)
    env["triple_dir"] = f'{env["peptide_bond_dir"]}/triple'
    env["single_dir"] = f'{env["peptide_bond_dir"]}/single'
    return env
