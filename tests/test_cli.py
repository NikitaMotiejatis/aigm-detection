import pytest

from aigm import __version__
from aigm.cli import main


def test_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "usage: aigm" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out
