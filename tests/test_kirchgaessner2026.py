"""Tests of the Kirchgässner (2026) equivalent circuits.

Classical (kirchgaessner2026) and MLP16-scheduled (kirchgaessner2026_mlp16).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from magnethub.sequence import SequenceModel, MATERIALS, TEAMS
from magnethub.siegen_paderborn_sequence import OutOfTrainingRangeWarning

TEAM_KEYS = ["kirchgaessner2026", "kirchgaessner2026_mlp16"]
TEST_FILES = Path(__file__).parent / "test_files"


def measured_3c90():
    """Measured 3C90 sequences of the shared test file: B, H (X, W) and T (X,)."""
    ds = pd.read_csv(TEST_FILES / "unit_test_data_sequence_3C90.csv")
    B = ds[[c for c in ds if c.startswith("B_t_")]].to_numpy()
    H = ds[[c for c in ds if c.startswith("H_t_")]].to_numpy()
    return B, H, ds["temp"].to_numpy()


@pytest.mark.parametrize("team", TEAM_KEYS)
def test_registered(team):
    """Both team keys are registered."""
    assert team in TEAMS


@pytest.mark.parametrize("team", TEAM_KEYS)
def test_reproduces_reference(team):
    """The NumPy circuits reproduce the trained PyTorch models (float64) on synthetic windows."""
    ref = json.loads((TEST_FILES / "kirchgaessner2026_reference.json").read_text())["materials"]
    for mat, d in ref.items():
        mdl = SequenceModel(material=mat, team=team)
        B, H, T = np.array(d["B"]), np.array(d["H"]), np.array(d["T"])
        for key, expected in d["expected"].items():
            if not key.startswith(team + "/"):
                continue
            k = int(key.split("/")[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OutOfTrainingRangeWarning)
                h = mdl(B[:, k:], T, B[:, :k], H[:, :k])
            expected = np.array(expected)
            assert np.abs(h - expected).max() < 1e-6 * np.abs(expected).max(), f"{team} {mat} warmup {k}"


@pytest.mark.parametrize("team", TEAM_KEYS)
def test_material_availability(team):
    """Every material loads and gives a finite H sequence of the right shape."""
    t = np.arange(320) / 16e6
    b = 0.1 * np.sin(2 * np.pi * 100e3 * t)
    for mat in MATERIALS:
        h = SequenceModel(material=mat, team=team)(b[64:], 50.0, b[:64], np.zeros(64))
        assert h.shape == (1, 256) and np.isfinite(h).all(), mat


@pytest.mark.parametrize("team", TEAM_KEYS)
def test_call_variants(team):
    """Single window, batch with per-window temperature, no warmup and any future length."""
    mdl = SequenceModel(material="N87", team=team)
    t = np.arange(600) / 16e6
    b = 0.1 * np.sin(2 * np.pi * np.array([[100e3], [200e3]]) * t)
    h_batch = mdl(b[:, 100:], np.array([25.0, 70.0]), b[:, :100], np.zeros((2, 100)))
    assert h_batch.shape == (2, 500)
    h_single = mdl(b[1, 100:], 70.0, b[1, :100], np.zeros(100))
    np.testing.assert_allclose(h_single[0], h_batch[1], rtol=1e-12, atol=1e-12)
    assert mdl(b[0, :500], 25.0).shape == (1, 500)                      # no warmup: starts from B = 0
    assert mdl(b[0, 100:117], 25.0, b[0, :100], np.zeros(100)).shape == (1, 17)


@pytest.mark.parametrize("team", TEAM_KEYS)
def test_accuracy_on_measured_data(team):
    """Same accuracy criteria as for the GRU models on the measured 3C90 sequences."""
    B, H, T = measured_3c90()
    warmup = 128
    h_pred = SequenceModel(material="3C90", team=team)(B[:, warmup:], T, B[:, :warmup], H[:, :warmup])
    h_true = H[:, warmup:]
    rel_l2 = np.linalg.norm(h_pred - h_true, axis=1) / np.linalg.norm(h_true, axis=1)
    assert rel_l2.max() < 0.15, f"per-row relative L2 errors: {rel_l2}"
    for i in range(len(B)):
        assert np.corrcoef(h_pred[i], h_true[i])[0, 1] > 0.95, f"low correlation for operating point {i}"


@pytest.mark.parametrize("team", ["kirchgaessner2026", "kirchgaessner2026_mlp16"])
def test_warns_outside_training_range(team):
    """Both circuits warn for temperatures, amplitudes and slopes beyond their training data."""
    mdl = SequenceModel(material="3C90", team=team)
    with pytest.warns(OutOfTrainingRangeWarning, match="T from"):
        mdl(np.zeros(16), 100.0)
    with pytest.warns(OutOfTrainingRangeWarning, match=r"\|B\| up to"):
        mdl(np.full(16, 1.0), 25.0)
    t = np.arange(320) / 16e6
    b = 0.4 * (1.0 - 4.0 * np.abs((t * 100e3 + 0.25) % 1.0 - 0.5))   # 0.4 T triangle at 100 kHz
    with pytest.warns(OutOfTrainingRangeWarning, match="dB/dt"):
        mdl(b, 25.0)


@pytest.mark.parametrize("team", ["kirchgaessner2026", "kirchgaessner2026_mlp16"])
def test_measured_data_does_not_warn(team):
    """Measured data within the training range raise no warning."""
    B, H, T = measured_3c90()
    with warnings.catch_warnings():
        warnings.simplefilter("error", OutOfTrainingRangeWarning)
        SequenceModel(material="3C90", team=team)(B[:, 128:], T, B[:, :128], H[:, :128])
