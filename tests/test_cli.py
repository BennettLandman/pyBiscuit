from pybiscuit.cli import main


def test_version(capsys):
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0
    assert "pybiscuit" in capsys.readouterr().out
