# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
import re
from unittest.mock import patch

import pytest

from vllm import version


def test_version_is_defined():
    assert version.__version__ is not None


def test_version_tuple():
    assert len(version.__version_tuple__) in (3, 4, 5)


@pytest.mark.parametrize(
    "version_tuple, version_str, expected",
    [
        ((0, 0, "dev"), "0.0", True),
        ((0, 0, "dev"), "foobar", True),
        ((0, 7, 4), "0.6", True),
        ((0, 7, 4), "0.5", False),
        ((0, 7, 4), "0.7", False),
        ((1, 2, 3), "1.1", True),
        ((1, 2, 3), "1.0", False),
        ((1, 2, 3), "1.2", False),
        # This won't work as expected
        ((1, 0, 0), "1.-1", True),
        ((1, 0, 0), "0.9", False),
        ((1, 0, 0), "0.17", False),
    ],
)
def test_prev_minor_version_was(version_tuple, version_str, expected):
    with patch("vllm.version.__version_tuple__", version_tuple):
        assert version._prev_minor_version_was(version_str) == expected


# ---- build_date tests ----
class TestBuildDate:
    """Tests for the __build_date__ field in vllm.version."""

    def test_build_date_is_exposed(self):
        """__build_date__ should be importable from vllm.version."""
        assert hasattr(version, "__build_date__")

    def test_build_date_is_string(self):
        """After a normal build, __build_date__ should be a non-None string."""
        assert version.__build_date__ is not None
        assert isinstance(version.__build_date__, str)

    def test_build_date_is_utc_iso8601(self):
        """__build_date__ must be a valid UTC ISO 8601 timestamp (+0000)."""
        bd = version.__build_date__
        assert bd is not None
        # Expected format: YYYY-MM-DDTHH:MM:SS+0000
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+0000", bd
        ), f"build_date '{bd}' does not match expected UTC ISO 8601 format"
        # Verify it is parseable and truly UTC
        parsed = datetime.datetime.strptime(bd, "%Y-%m-%dT%H:%M:%S%z")
        assert parsed.utcoffset() == datetime.timedelta(0)

    def test_build_date_fallback_when_missing(self):
        """version.__build_date__ should be None when _version.py lacks it."""
        with patch("vllm.version.__build_date__", None):
            assert version.__build_date__ is None

    def test_build_date_in_init(self):
        """__build_date__ should be re-exported from vllm.__init__."""
        import vllm
        assert hasattr(vllm, "__build_date__")
        assert vllm.__build_date__ == version.__build_date__
