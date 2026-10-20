import re

import pytest

from conftest import OPT
from pybiscuit import assets


def test_registry_loads_and_schema():
    reg = assets.load_registry()
    assert len(reg) > 0
    for rel, e in reg.items():
        assert rel == e.path
        assert not rel.startswith("/") and "\\" not in rel and ".." not in rel
        assert rel.split("/")[0] in {"reg", "parc", "sphere"}
        assert isinstance(e.size, int) and e.size > 0
        assert re.fullmatch(r"[0-9a-f]{64}", e.sha256)
    assert list(reg) == sorted(reg)


def test_registry_expected_contents():
    reg = assets.load_registry()
    assert "sphere/sphere_327680.vtk" in reg
    assert "reg/mb101/lh.white.vtk" in reg
    assert "parc/dkt31/lh.pth" in reg


def test_status_summary(monkeypatch, tmp_path):
    monkeypatch.setenv(assets.CACHE_ENV, str(tmp_path))
    s = assets.status_summary()
    n = len(assets.load_registry())
    assert s.startswith(f"assets: {n} files registered, 0 present in cache")


def test_verify_detects_problems(tmp_path):
    reg = assets.load_registry()
    rep = assets.verify(tmp_path)
    assert len(rep.missing) == len(reg) and not rep.passed
    # fabricate one file with the wrong size and one with right size, wrong hash
    smallest = sorted(reg.values(), key=lambda e: e.size)[:2]
    a, b = smallest
    for e, size in ((a, a.size + 1), (b, b.size)):
        p = tmp_path.joinpath(*e.path.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
    rep = assets.verify(tmp_path)
    assert a.path in rep.size_mismatch
    assert b.path in rep.hash_mismatch
    assert rep.ok == []


@pytest.mark.reference
def test_registry_matches_reference():
    reg = assets.load_registry()
    on_disk = {
        p.relative_to(OPT).as_posix()
        for sub in ("reg", "parc", "sphere")
        for p in (OPT / sub).rglob("*")
        if p.is_file()
    }
    assert on_disk == set(reg)
    for rel, e in reg.items():
        assert (OPT / rel).stat().st_size == e.size, rel
    rep = assets.verify(OPT)
    assert rep.passed, (rep.missing, rep.size_mismatch, rep.hash_mismatch)
    assert len(rep.ok) == len(reg)
