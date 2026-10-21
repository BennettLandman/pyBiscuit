"""Mock end-to-end test of the image reference harness (brief 10, B1.1/B1.2/B1.4).

A fake "image" (a directory ``opt/{biscuit,cmorph}`` of stub executables) and an ``apptainer``
stub on ``PATH`` stand in for ``biscuit_FC_v2.2.sif``; the real image is never run.  The stub maps
bind-mount destinations (``/INPUTS``, ``/harness``, ...) and ``/opt`` onto host directories, which
is enough to exercise the PATH shims, JSONL logging, ``trace_to_spec.py``, the full-run archive and
the golden writer on macOS or Linux.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "reference"
PY = sys.executable

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash harness")

APPTAINER_STUB = r'''#!/usr/bin/env python3
"""Stand-in for `apptainer exec`: bind destinations and /opt are remapped onto host paths."""
import os, subprocess, sys, tempfile
from pathlib import Path

IMG = Path(os.environ["MOCK_IMAGE_ROOT"])
a = sys.argv[1:]
if a[:1] == ["--version"]:
    print("apptainer version mock-1.0")
    sys.exit(0)
assert a[0] == "exec", a
a = a[1:]
binds, envs, pwd = {}, {}, None
i = 0
while i < len(a) and a[i].startswith("-"):
    o = a[i]
    if o == "--cleanenv":
        i += 1
    elif o in ("--home",):
        i += 2
    elif o == "--pwd":
        pwd = a[i + 1]; i += 2
    elif o == "-B":
        src, dst = a[i + 1].split(":")[:2]; binds[dst] = src; i += 2
    elif o == "--env":
        k, _, v = a[i + 1].partition("="); envs[k] = v; i += 2
    else:
        raise SystemExit(f"stub: unsupported option {o}")
sif, cmd = a[i], a[i + 1:]
assert Path(sif).is_file(), sif


def remap(s):
    for dst in sorted(binds, key=len, reverse=True):
        s = s.replace(dst + "/", binds[dst] + "/").replace(dst, binds[dst]) if dst in s else s
    return s.replace("/opt/", str(IMG / "opt") + "/")


env = dict(os.environ)
env.update({k: remap(v) for k, v in envs.items()})
opt = IMG / "opt"
env["PATH"] = ":".join([str(opt / "biscuit"), str(opt / "biscuit" / "bin"), str(opt / "cmorph" / "bin"), env["PATH"]])
env["BISCUIT_ROOT"] = str(opt / "biscuit")
env["BISCUIT_SHIM_REAL_PATH"] = f"{opt / 'biscuit' / 'bin'}:{opt / 'cmorph' / 'bin'}"
args = [remap(c) for c in cmd]
if args[0] == "bash" and Path(args[1]).is_file():  # rewrite container paths inside the script text
    text = remap(Path(args[1]).read_text())
    tmp = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)
    tmp.write(text); tmp.close()
    args[1] = tmp.name
sys.exit(subprocess.run(args, env=env, cwd=remap(pwd) if pwd else None).returncode)
'''

RUN_BISCUIT = r"""#!/bin/bash
# fake run_biscuit: same cleanup lines as the real one, three sub-tool calls
while [[ $# -gt 0 ]]; do case $1 in --reg|--surf_recon|--thread) shift 2;; *) POS+=("$1"); shift;; esac; done
IN=${POS[0]}; OUT=${POS[1]}
export TMP=$OUT/.tmp
mkdir -p $TMP; CONF=$TMP/biscuit.conf; echo "reg_temp = x" > $CONF
cd $TMP
cp $IN/lh.white.vtk surf.vtk
MeshProperty -i surf.vtk -o surf -m --writevtk || exit 1
lgi surf.vtk
HSD -s surf.vtk -o "surf reg.vtk" -d 4 --note 'quo"te'
cd $OUT
mkdir -p pdf
rm $CONF
rm -rf $TMP
"""

LGI = r"""#!/bin/bash
rp=$(dirname "$0")
"$rp/CurveExtraction" -i "$1" -o surf --sulcus --gyrus
"$rp/Gyrification" -i "$1" -o lgi -s surf.scurve -g surf.gcurve
"""

# stub tools: write the outputs named by their flags (and a failing tool for exit-code checks)
TOOLS = {
    "MeshProperty": 'echo "mp $*"; out=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done; echo vtk > "$out.property.vtk"',
    "CurveExtraction": 'echo "ce $*"; echo s > surf.scurve; echo g > surf.gcurve',
    "Gyrification": 'echo "gy $*"; test -s surf.scurve && echo 1.0 > lgi.lgi.map.1.txt',
    "HSD": 'echo "hsd $*" >&2; out=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done; echo reg > "$out"',
    "SphericalRemesh": "echo sr; echo v > out.vtk; echo p > outprop.prop.txt; echo b > bary.txt",
    "FailTool": "echo bad >&2; exit 3",
}


def write_exe(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture()
def mock(tmp_path):
    img = tmp_path / "image"
    for name, body in TOOLS.items():
        d = "cmorph" if name in ("CurveExtraction", "Gyrification", "HSD", "SphericalRemesh") else "biscuit"
        write_exe(img / "opt" / d / "bin" / name, f"#!/bin/bash\n{body}\n")
    write_exe(img / "opt" / "cmorph" / "bin" / "lgi", LGI)
    write_exe(img / "opt" / "biscuit" / "run_biscuit", RUN_BISCUIT)
    bindir = tmp_path / "bin"
    write_exe(bindir / "apptainer", APPTAINER_STUB.replace("#!/usr/bin/env python3", f"#!{PY}"))
    sif = tmp_path / "mock.sif"
    sif.write_bytes(b"not a real image")
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "lh.white.vtk").write_text("mesh\n")
    env = dict(os.environ, MOCK_IMAGE_ROOT=str(img), APPTAINER=str(bindir / "apptainer"))
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    return {"img": img, "sif": sif, "in": inp, "env": env, "tmp": tmp_path, "bin": bindir}


def sh(args, env, **kw):
    return subprocess.run([str(a) for a in args], env=env, capture_output=True, text=True, **kw)


def test_scripts_dry_run_print_commands(tmp_path):
    r = sh([SCRIPTS / "run_full.sh", "--sif", "/x/FC.sif", "--input", "/in", "--out", tmp_path / "o",
            "--license", "/lic/license.txt", "--reg", "oasis45", "--dry-run"], os.environ)
    assert r.returncode == 0, r.stderr
    assert "apptainer exec --cleanenv" in r.stdout
    assert "/lic/license.txt:/opt/biscuit/license.txt:ro" in r.stdout
    assert "REG=oasis45" in r.stdout and "SURF_RECON=macruise" in r.stdout
    assert not (tmp_path / "o.harness").exists(), "dry run must not touch the filesystem"
    r = sh([SCRIPTS / "install_shims.sh", "--dry-run", tmp_path / "s"], os.environ)
    assert r.returncode == 0 and "ln -sf trace_shim.sh" in r.stdout and not (tmp_path / "s").exists()
    r = sh([PY, SCRIPTS / "make_image_golden.py", "--sif", "/x.sif", "--tool", "meshproperty",
            "--case", "all_maps", "--dry-run"], os.environ)
    assert r.returncode == 0, r.stderr
    assert "MeshProperty -i bumpy.vtk -o bumpy -c -s -k -m -v --writevtk" in r.stdout


def test_shim_logs_exit_code_and_files(mock, tmp_path):
    shims, trace = tmp_path / "shims", tmp_path / "trace"
    assert sh([SCRIPTS / "install_shims.sh", shims, "FailTool", "HSD"], mock["env"]).returncode == 0
    work = tmp_path / "w"
    work.mkdir()
    env = dict(mock["env"], BISCUIT_TRACE_DIR=str(trace), BISCUIT_SHIM_REAL_PATH=str(mock["img"] / "opt/cmorph/bin") + ":" + str(mock["img"] / "opt/biscuit/bin"), OMP_NUM_THREADS="1")
    r = sh([shims / "FailTool", "a b", 'q"x'], env, cwd=work)
    assert r.returncode == 3 and "bad" in r.stderr
    r = sh([shims / "HSD", "-o", "reg.vtk", "--note", "multi\nline"], env, cwd=work)
    assert r.returncode == 0 and "hsd" in r.stderr
    recs = [json.loads(p.read_text()) for p in sorted(trace.glob("*.json"))]
    by = {x["tool"]: x for x in recs}
    assert by["FailTool"]["exit"] == 3 and by["FailTool"]["argv"] == ["a b", 'q"x']
    assert by["HSD"]["env"] == {"OMP_NUM_THREADS": "1"}
    assert by["HSD"]["argv"][-1] == "multi\\nline".replace("\\n", "\n")
    assert any(c.endswith("reg.vtk") for c in by["HSD"]["created"])
    assert by["HSD"]["end"] >= by["HSD"]["start"] and by["HSD"]["cwd"] == str(work)
    assert (trace / by["HSD"]["stderr"]).read_text().startswith("hsd")
    missing = sh([shims / "HSD"], dict(env, BISCUIT_SHIM_REAL_PATH="/nonexistent"), cwd=work)
    assert missing.returncode == 127


def test_run_full_trace_spec_and_manifest(mock, tmp_path):
    out = tmp_path / "out" / "subj"
    r = sh([SCRIPTS / "run_full.sh", "--sif", mock["sif"], "--input", mock["in"], "--out", out,
            "--reg", "oasis45", "--no-verify"], mock["env"])
    assert r.returncode == 0, r.stdout + r.stderr
    work = Path(str(out) + ".harness")
    assert (out / ".tmp" / "biscuit.conf").exists(), "cleanup lines must be removed"
    assert (out / "pdf").is_dir()
    recs = [json.loads(x) for x in (work / "trace.jsonl").read_text().splitlines()]
    tools = [x["tool"] for x in recs]
    assert tools == ["MeshProperty", "CurveExtraction", "Gyrification", "HSD"], "lgi tools must be traced"
    assert all(x["exit"] == 0 for x in recs)
    run = json.loads((work / "run.json").read_text())
    assert run["reg"] == "oasis45" and run["exit"] == 0 and run["trace"] == 1
    assert len(run["sif_sha256"]) == 64
    manifest = (work / "MANIFEST.sha256").read_text().splitlines()
    assert any(line.endswith("subj/.tmp/biscuit.conf") for line in manifest)
    assert (work / "subj.tar.gz").stat().st_size > 0

    md_path = tmp_path / "spec.md"
    r = sh([PY, SCRIPTS / "trace_to_spec.py", work / "trace.jsonl", "-o", md_path], mock["env"])
    assert r.returncode == 0, r.stderr
    md = md_path.read_text()
    assert "## Call sequence" in md and "### 3. Gyrification" in md
    flow = md.split("## File flow")[1].split("## Flags")[0]
    assert "CurveExtraction" in flow and "surf.scurve" in flow and "Gyrification" in flow
    assert "| `--sulcus`" in md or "`-m`" in md
    r = sh([PY, SCRIPTS / "trace_to_spec.py", work / "trace"], mock["env"])  # directory input
    assert r.returncode == 0 and "4 sub-tool calls" in r.stdout


def test_run_full_failure_still_archives(mock, tmp_path):
    write_exe(mock["img"] / "opt" / "biscuit" / "bin" / "MeshProperty", "#!/bin/bash\nexit 7\n")
    out = tmp_path / "out2"
    r = sh([SCRIPTS / "run_full.sh", "--sif", mock["sif"], "--input", mock["in"], "--out", out,
            "--no-verify"], mock["env"])
    assert r.returncode != 0
    work = Path(str(out) + ".harness")
    assert (work / "out2.tar.gz").exists() and json.loads((work / "run.json").read_text())["exit"] != 0


def test_make_image_golden_end_to_end(mock, tmp_path):
    gold = tmp_path / "golden-image"
    base = [PY, SCRIPTS / "make_image_golden.py", "--sif", mock["sif"], "--out", gold,
            "--apptainer", mock["bin"] / "apptainer", "--no-verify"]
    r = sh([*base, "--tool", "spherical-remesh", "--case", "up3to4"], mock["env"])
    assert r.returncode == 0, r.stdout + r.stderr
    d = gold / "spherical-remesh" / "up3to4"
    meta = json.loads((d / "command.json").read_text())
    assert meta["oracle"] == "image-biscuit_FC_v2.2"
    assert meta["image_sha256"] == __import__("hashlib").sha256(mock["sif"].read_bytes()).hexdigest()
    assert meta["argv"][0] == "SphericalRemesh" and "$ORACLES" not in json.dumps(meta)
    assert sorted(p.name for p in d.iterdir()) == ["bary.txt", "command.json", "out.vtk", "outprop.prop.txt"]
    r = sh([*base, "--tool", "hsd", "--case", "rot8_ico4"], mock["env"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (gold / "hsd" / "rot8_ico4" / "s1.reg.vtk").exists()
    r = sh([*base, "--tool", "meshdeform", "--case", "translate"], mock["env"])  # tool absent in mock
    assert r.returncode == 1 and "FAILED" in r.stderr

    sys.path.insert(0, str(REPO / "tests"))
    from _golden import golden_dir, golden_source

    root = tmp_path / "data"
    (root / "golden" / "t" / "c").mkdir(parents=True)
    assert golden_dir("t", "c", root=root) == root / "golden" / "t" / "c"
    (root / "golden-image" / "t" / "c").mkdir(parents=True)
    (root / "golden-image" / "t" / "c" / "command.json").write_text('{"oracle": "image-biscuit_FC_v2.2"}')
    assert golden_dir("t", "c", root=root) == root / "golden-image" / "t" / "c"
    assert golden_source("t", "c", root=root) == "image" and golden_source("t", "zz", root=root) == "missing"


def test_workflow_is_dispatch_only():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((REPO / ".github" / "workflows" / "reference.yml").read_text())
    triggers = wf.get("on", wf.get(True))
    assert list(triggers) == ["workflow_dispatch"]
    text = (REPO / ".github" / "workflows" / "reference.yml").read_text()
    assert "git add" not in text and "git push" not in text and "upload-artifact" in text
    assert textwrap.dedent(text).count("secrets.") >= 1
