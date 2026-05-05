from hydrolysis.utils import Error, check_logfiles_for_errors, pushd, LogCheckResult


def test_logfile_checks_for_errors():
    with pushd("src/tests/test_files"):
        job = "running"
        logcheck = check_logfiles_for_errors(job)
        assert logcheck == LogCheckResult(
            found_error=False,
            error_type=None,
            error_msg=None,
            latest_time=0.999,
            latest_distance=0.294847,
        )

        job = "converged"
        logcheck = check_logfiles_for_errors(job)
        assert logcheck == LogCheckResult(
            found_error=True,
            error_type=Error.GmxError,
            error_msg="Fatal error:\nPull reference distance for coordinate 3 (-0.000104) needs to be non-negative\n\n",
            latest_time=1.9135,
            latest_distance=0.133538,
        )
