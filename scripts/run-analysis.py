from plotnine import *  # pyright: ignore
from hamilton import driver
import hydrolysis.analysis as analysis
from hydrolysis.constants import HAMILTON_CACHE

import logging
logging.basicConfig(level=logging.INFO)

def main():
    dr = (
        driver.Builder()
        .with_modules(analysis)
        .with_cache(
            path=HAMILTON_CACHE,
        )
        .build()
    )

    final_vars = [
        "envs",
        "parameters",
        "frame_cwds",
        "example_equilibration",
        "wethyd_us_energies",
        "parameters",
        "wethyd_choices",
        "wethyd_us_hist",
        "wethyd_us_prof",
        "wethyd_us_comb",
        "wetbreak_us",
        "wetbreak_us_comb",
        "break_us_distances",
        "break_us_protons"
    ]

    result = dr.execute(
      final_vars=final_vars # pyright: ignore
    )


if __name__ == "__main__":
    main()
