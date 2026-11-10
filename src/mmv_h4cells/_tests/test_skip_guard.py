"""Tests for the --fail-on-skip option defined in the root conftest.py"""

from pathlib import Path

import pytest

ROOT_CONFTEST = Path(__file__).parents[3] / "conftest.py"

TESTS_WITH_SKIP = """
import pytest

def test_passes():
    pass

@pytest.mark.skip("not needed")
def test_skipped():
    pass
"""

TESTS_WITHOUT_SKIP = """
def test_passes():
    pass
"""


@pytest.fixture
def pytester_with_guard(pytester):
    pytester.makeconftest(ROOT_CONFTEST.read_text())
    return pytester


def test_skipped_test_fails_the_run_with_the_flag(pytester_with_guard):
    pytester_with_guard.makepyfile(TESTS_WITH_SKIP)

    result = pytester_with_guard.runpytest("--fail-on-skip")

    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.assert_outcomes(passed=1, skipped=1)
    result.stdout.fnmatch_lines(["*1 test(s) were skipped*"])


def test_skipped_test_is_allowed_without_the_flag(pytester_with_guard):
    pytester_with_guard.makepyfile(TESTS_WITH_SKIP)

    result = pytester_with_guard.runpytest()

    assert result.ret == pytest.ExitCode.OK


def test_run_without_skips_passes_with_the_flag(pytester_with_guard):
    pytester_with_guard.makepyfile(TESTS_WITHOUT_SKIP)

    result = pytester_with_guard.runpytest("--fail-on-skip")

    assert result.ret == pytest.ExitCode.OK
