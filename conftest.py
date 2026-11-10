"""Test configuration shared by all tests."""

import pytest

pytest_plugins = ["pytester"]  # used to test the --fail-on-skip guard


def pytest_addoption(parser):
    parser.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help=(
            "Fail the test run if any test was skipped. Used in CI so that "
            "skipped tests cannot pile up unnoticed."
        ),
    )


def pytest_sessionfinish(session, exitstatus):
    if not session.config.getoption("--fail-on-skip"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if skipped and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter.write_line(
            f"ERROR: {len(skipped)} test(s) were skipped (--fail-on-skip)",
            red=True,
        )
