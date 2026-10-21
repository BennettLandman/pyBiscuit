"""Parcellation package pieces that must work without PyTorch (core CI job)."""

import collections
import os
import pickle

import numpy as np
import pytest

from pybiscuit.parc import fusion
from pybiscuit.parc.checkpoint import UnsafeCheckpointError, audit_legacy_pickle


class _Boom:
    """Pickle that would run ``os.system`` if it were ever unpickled."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def _legacy(path, main_obj, main_protocol=2):
    """Write a file with the legacy layout: magic, version, sys info, state, storage keys."""
    with open(path, "wb") as fh:
        pickle.dump(0x1950A86A20F9469CFC6C, fh, protocol=2)
        pickle.dump(1001, fh, protocol=2)
        pickle.dump({"little_endian": True}, fh, protocol=2)
        pickle.dump(main_obj, fh, protocol=main_protocol)
        pickle.dump([], fh, protocol=2)
    return path


def test_audit_accepts_allowed_globals(tmp_path):
    p = _legacy(tmp_path / "ok.pth", {"state_dict": collections.OrderedDict()})
    assert audit_legacy_pickle(p)["globals"] == {"collections.OrderedDict": 1}


def test_audit_rejects_payload_without_executing_it(tmp_path):
    marker = tmp_path / "pwned"
    p = _legacy(tmp_path / "evil.pth", {"state_dict": _Boom(marker)})
    with pytest.raises(UnsafeCheckpointError, match="posix.system|os.system|not allowed"):
        audit_legacy_pickle(p)
    assert not marker.exists()


def test_audit_rejects_stack_global_and_non_legacy(tmp_path):
    marker = tmp_path / "pwned"
    p = _legacy(tmp_path / "evil4.pth", {"state_dict": _Boom(marker)}, main_protocol=4)
    with pytest.raises(UnsafeCheckpointError, match="STACK_GLOBAL|not allowed"):
        audit_legacy_pickle(p)
    zipish = tmp_path / "new.pth"
    zipish.write_bytes(b"PK\x03\x04" + b"\0" * 64)
    with pytest.raises(UnsafeCheckpointError, match="not a legacy"):
        audit_legacy_pickle(zipish)
    assert not marker.exists()


def test_fusion_rule_is_pending():
    outs = [np.zeros((3, 4)), np.ones((3, 4))]
    with pytest.raises(NotImplementedError, match="pending B7.1"):
        fusion.fuse(outs)
    np.testing.assert_array_equal(fusion.fuse(outs, rule=lambda a: np.mean(a, axis=0)), 0.5)
    with pytest.raises(ValueError):
        fusion.fuse([np.zeros((3, 4)), np.zeros((2, 4))], rule=lambda a: a[0])


def test_package_import_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys, pybiscuit.parc, pybiscuit.parc.fusion, pybiscuit.parc.checkpoint;"
    code += "sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
