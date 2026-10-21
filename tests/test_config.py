import pytest

from conftest import OPT
from pybiscuit.config import (
    CLI_FLAGS,
    DEFAULTS,
    Config,
    ConfigError,
    load_config,
    read_conf,
)

SHIPPED = OPT / "biscuit.conf"


def test_defaults_are_complete_and_typed():
    cfg = load_config()
    assert cfg["reg_temp"] == "mb101"
    assert cfg["parc_protocol"] == ("braincolor", "dkt31")
    assert cfg["trace_delta"] == 2.5 and cfg["lgi_kernel_size"] == 316
    assert set(CLI_FLAGS.values()) <= set(DEFAULTS)


def test_conf_reader_and_writer(tmp_path):
    p = tmp_path / "biscuit.conf"
    p.write_text("% c\nproject = X \n\nreg_temp=oasis45\nparc_protocol = braincolor, dkt31\n")
    cfg = load_config(conf=p)
    assert cfg["project"] == "X" and cfg["reg_temp"] == "oasis45"
    assert cfg["parc_protocol"] == ("braincolor", "dkt31")
    assert load_config(conf=_write(tmp_path, cfg.to_conf())).values == cfg.values


def _write(tmp_path, text):
    q = tmp_path / "rt.conf"
    q.write_text(text)
    return q


@pytest.mark.reference
def test_roundtrip_shipped_conf(tmp_path):
    raw = read_conf(SHIPPED)
    assert list(raw) == list(DEFAULTS)  # every key, same order
    cfg = load_config(conf=SHIPPED)
    assert cfg.values == load_config().values  # defaults == FC image defaults
    assert cfg.to_conf().strip() == SHIPPED.read_text().strip()  # byte layout identical
    assert load_config(conf=_write(tmp_path, cfg.to_conf())).values == cfg.values


def test_prefix_filtering():
    cfg = load_config(overrides={"surf_recon": "MaCRUISE"})
    assert cfg["surf_recon"] == "macruise"  # lower-cased as in run_biscuit
    keys = cfg.active_items()
    assert "mc_surf" in keys and not any(k.startswith(("fs_", "cv_")) for k in keys)
    text = cfg.to_conf(active_only=True)
    assert "% macruise" in text and "% freesurfer" not in text and "% civet" not in text
    assert "fs_" not in load_config(overrides={"surf_recon": "civet"}).to_conf(active_only=True)


def test_cli_override_names_and_precedence(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text('[biscuit]\nreg_temp = "oasis45"\ntrace_r = 5\nparc_protocol = ["dkt31"]\n')
    conf = tmp_path / "b.conf"
    conf.write_text("reg_temp = mb101\nsci_bins = 7\n")
    over = {f: None for f in CLI_FLAGS}
    over.update(reg="mb101", lgi_aniso="0.5", thread="4", subj="s01")
    cfg = load_config(conf=conf, toml=toml, overrides=over)
    assert cfg["reg_temp"] == "mb101"  # CLI beats TOML beats conf
    assert cfg["trace_r"] == 5 and cfg["sci_bins"] == 7  # TOML / conf survive
    assert cfg["parc_protocol"] == ("dkt31",)
    assert cfg["lgi_anisotropy"] == 0.5 and cfg["omp_num_threads"] == 4
    assert cfg["subject"] == "s01"
    assert cfg.sources[0] == "defaults" and cfg.sources[-1].startswith("overrides:")


def test_flat_toml(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('surf_smooth = 3\nsurf_recon = "macruise"\n')
    cfg = load_config(toml=p)
    assert cfg["surf_smooth"] == 3 and cfg["surf_recon"] == "macruise"


@pytest.mark.parametrize(
    "over",
    [{"surf_recon": "bogus"}, {"trace_r": "x"}, {"sci_bins": 2.5}, {"nope": 1}, {"parc": ""}],
)
def test_invalid_values(over):
    with pytest.raises(ConfigError):
        load_config(overrides=over)


def test_unknown_conf_key_and_syntax(tmp_path):
    p = tmp_path / "b.conf"
    p.write_text("bogus = 1\n")
    with pytest.raises(ConfigError, match="unknown configuration key"):
        read_conf(p)
    p.write_text("no equals sign\n")
    with pytest.raises(ConfigError, match="key = value"):
        read_conf(p)


def test_replace_and_dict():
    cfg = load_config().replace(trace_r="6")
    assert isinstance(cfg, Config) and cfg["trace_r"] == 6
    assert cfg.to_dict()["parc_protocol"] == ["braincolor", "dkt31"]
