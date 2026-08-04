"""Tests for the optional-dependency helper."""

import pytest

from humanizer_bench._deps import require


def test_require_returns_installed_module():
    json_module = require("json")
    assert json_module.dumps({"a": 1}) == '{"a": 1}'


def test_require_missing_module_names_the_install_command():
    with pytest.raises(ImportError) as excinfo:
        require("definitely_not_installed_xyz")
    message = str(excinfo.value)
    assert "definitely_not_installed_xyz" in message
    assert 'pip install -e ".[models]"' in message


def test_require_reports_the_requested_extra():
    with pytest.raises(ImportError, match=r"\[dev\]"):
        require("definitely_not_installed_xyz", extra="dev")
