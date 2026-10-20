from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO.parent / "reference"
OPT = REFERENCE / "sif_FC_v2.2" / "root" / "opt" / "biscuit"
CTF_PLAIN = REFERENCE / "ctf_plain"
REPOS = REFERENCE / "repos"
PARC_MESH_FILES = CTF_PLAIN / "SurfPostproc" / "common" / "parc" / "mesh_files"
SPHERE_DIR = OPT / "sphere"
REG_DIR = OPT / "reg"
MIPAV = Path.home() / "dev" / "mipav"


def pytest_collection_modifyitems(config, items):
    missing = not (OPT.exists() and MIPAV.exists())
    skip = pytest.mark.skip(reason="reference/ or ~/dev/mipav not available")
    run_bitexact = "bitexact" in (config.getoption("-m") or "")
    skip_bx = pytest.mark.skip(reason="bit-exact diagnostics: run with -m bitexact")
    for item in items:
        if "bitexact" in item.keywords and not run_bitexact:
            item.add_marker(skip_bx)
        if "reference" in item.keywords and missing:
            item.add_marker(skip)
