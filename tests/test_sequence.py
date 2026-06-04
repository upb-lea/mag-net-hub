import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from magnethub.sequence import SequenceModel, MATERIALS, TEAMS


class TestSequenceModelValidation:
    """Tests for input validation and error handling at construction time."""

    def test_invalid_material_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            SequenceModel(material="INVALID", team="paderborn")

    def test_invalid_team_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            SequenceModel(material="3C92", team="nonexistent")

    def test_paderborn_is_registered(self):
        assert "paderborn" in TEAMS, "Expected the paderborn team to be registered"

    def test_case_insensitive_material(self):
        """Material should be uppercased internally."""
        mdl = SequenceModel(material="3c90", team="paderborn")
        assert mdl.material == "3C90"

    def test_case_insensitive_team(self):
        """Team should be lowercased internally."""
        mdl = SequenceModel(material="3C90", team="PADERBORN")
        assert mdl.team == "paderborn"


@pytest.fixture(scope="module")
def model():
    """A real Paderborn sequence model for 3C90, shared across tests in a module."""
    return SequenceModel(material="3C90", team="paderborn")


class TestSequenceModelCall:
    """Tests for the __call__ interface using a real model backend."""

    def test_single_waveform(self, model):
        warmup, future = 100, 400
        b_past = np.random.randn(warmup) * 200e-3
        h_past = np.random.randn(warmup) * 5
        b_future = np.random.randn(future) * 200e-3

        h = model(b_future, 50.0, b_past, h_past)

        assert h.shape == (1, future)
        assert np.isfinite(h).all()

    def test_batch_waveform(self, model):
        warmup, future, batch = 80, 256, 10
        b_past = np.random.randn(batch, warmup) * 200e-3
        h_past = np.random.randn(batch, warmup) * 5
        b_future = np.random.randn(batch, future) * 200e-3
        temps = np.full(batch, 60.0)

        h = model(b_future, temps, b_past, h_past)

        assert h.shape == (batch, future)
        assert np.isfinite(h).all()

    def test_variable_future_length_accepted(self, model):
        """Any future sequence length should be accepted without resampling."""
        warmup = 64
        for length in [50, 233, 1024, 2048]:
            b_past = np.random.randn(1, warmup) * 200e-3
            h_past = np.random.randn(1, warmup) * 5
            b_future = np.random.randn(1, length) * 200e-3
            h = model(b_future, 50.0, b_past, h_past)
            assert h.shape == (1, length), f"Failed for future length {length}"

    def test_minimal_warmup_window(self, model):
        """A single-sample warmup window (warmup is skipped internally) is valid."""
        b_past = np.random.randn(3, 1) * 200e-3
        h_past = np.zeros((3, 1))
        b_future = np.random.randn(3, 300) * 200e-3
        h = model(b_future, np.array([25.0, 40.0, 60.0]), b_past, h_past)
        assert h.shape == (3, 300)
        assert np.isfinite(h).all()

    def test_scalar_past(self, model):
        """Scalar b_past / h_past (single value, no warmup) should be broadcast."""
        b_future = np.random.randn(256) * 200e-3
        h = model(b_future, 50.0, b_past=0.0, h_past=0.0)
        assert h.shape == (1, 256)
        assert np.isfinite(h).all()

    def test_no_past_arguments(self, model):
        """Omitting b_past and h_past entirely should default to zeros."""
        b_future = np.random.randn(3, 256) * 200e-3
        h = model(b_future, np.array([25.0, 50.0, 75.0]))
        assert h.shape == (3, 256)
        assert np.isfinite(h).all()

    def test_repeatability(self, model):
        """The same input must yield the same output."""
        warmup, future = 128, 512
        b_past = np.random.randn(2, warmup) * 200e-3
        h_past = np.random.randn(2, warmup) * 5
        b_future = np.random.randn(2, future) * 200e-3
        temps = np.array([30.0, 70.0])

        h1 = model(b_future, temps, b_past, h_past)
        h2 = model(b_future, temps, b_past, h_past)
        assert np.allclose(h1, h2)

    def test_scalar_temperature_broadcast(self, model):
        """A scalar temperature is wrapped to a 1-D array of the batch size."""
        b_past = np.random.randn(1, 50) * 200e-3
        h_past = np.zeros((1, 50))
        b_future = np.random.randn(1, 128) * 200e-3
        h = model(b_future, 42.0, b_past, h_past)
        assert h.shape == (1, 128)

    def test_batch_size_mismatch_raises(self, model):
        b_past = np.random.randn(3, 50) * 200e-3
        h_past = np.random.randn(3, 50) * 5
        b_future = np.random.randn(2, 128) * 200e-3
        with pytest.raises(AssertionError, match="Batch sizes disagree"):
            model(b_future, np.array([25.0, 30.0, 35.0]), b_past, h_past)

    def test_warmup_length_mismatch_raises(self, model):
        b_past = np.random.randn(2, 50) * 200e-3
        h_past = np.random.randn(2, 40) * 5
        b_future = np.random.randn(2, 128) * 200e-3
        with pytest.raises(AssertionError, match="Warmup lengths disagree"):
            model(b_future, np.array([25.0, 30.0]), b_past, h_past)


class TestSequenceModelBackend:
    """Tests for the output-shape guards using a stubbed backend."""

    @pytest.fixture()
    def stub_model(self):
        """A SequenceModel with a fake inner backend, bypassing __init__."""
        mdl = object.__new__(SequenceModel)
        mdl.material = "3C90"
        mdl.team = "paderborn"
        mdl.mdl = lambda bp, hp, bf, t: np.zeros_like(bf)
        return mdl

    def test_inner_backend_receives_reshaped_args(self, stub_model):
        captured = {}

        def spy(bp, hp, bf, t):
            captured.update(b_past=bp, h_past=hp, b_future=bf, temp=t)
            return np.zeros_like(bf)

        stub_model.mdl = spy
        stub_model(np.random.randn(200) * 0.1, 42.0, np.random.randn(60) * 0.1, np.zeros(60))

        assert captured["b_past"].shape == (1, 60)
        assert captured["h_past"].shape == (1, 60)
        assert captured["b_future"].shape == (1, 200)
        np.testing.assert_array_equal(captured["temp"], np.array([42.0]))

    def test_output_shape_mismatch_raises(self, stub_model):
        stub_model.mdl = lambda bp, hp, bf, t: np.zeros((bf.shape[0], bf.shape[1] + 1))
        with pytest.raises(AssertionError, match="does not match"):
            stub_model(np.zeros((1, 128)), 50.0, np.zeros((1, 50)), np.zeros((1, 50)))

    def test_output_ndim_mismatch_raises(self, stub_model):
        stub_model.mdl = lambda bp, hp, bf, t: np.zeros(bf.shape[1])
        with pytest.raises(AssertionError, match="ndim"):
            stub_model(np.zeros((1, 128)), 50.0, np.zeros((1, 50)), np.zeros((1, 50)))


class TestSequenceModelAccuracy:
    """End-to-end accuracy check against measured 3C90 data."""

    def test_accuracy_on_measured_data(self, model):
        test_ds = pd.read_csv(Path(__file__).parent / "test_files" / "unit_test_data_sequence_3C90.csv")
        b_cols = [c for c in test_ds if c.startswith("B_t_")]
        h_cols = [c for c in test_ds if c.startswith("H_t_")]
        B = test_ds[b_cols].to_numpy()
        H = test_ds[h_cols].to_numpy()
        T = test_ds["temp"].to_numpy()

        warmup = 128
        h_pred = model(B[:, warmup:], T, B[:, :warmup], H[:, :warmup])
        h_true = H[:, warmup:]

        assert h_pred.shape == h_true.shape
        assert np.isfinite(h_pred).all()

        rel_l2 = np.linalg.norm(h_pred - h_true, axis=1) / np.linalg.norm(h_true, axis=1)
        assert rel_l2.max() < 0.15, f"Inaccurate H prediction, per-row relative L2 errors: {rel_l2}"

        for i in range(len(B)):
            corr = np.corrcoef(h_pred[i], h_true[i])[0, 1]
            assert corr > 0.95, f"Low correlation {corr:.4f} for operating point {i}"


def test_material_availability():
    """Every advertised material must load and produce a finite H sequence."""
    warmup, future = 64, 256
    b_past = np.random.randn(warmup) * 200e-3
    h_past = np.zeros(warmup)
    b_future = np.random.randn(future) * 200e-3
    temp = 50.0

    for m_lbl in MATERIALS:
        mdl = SequenceModel(material=m_lbl, team="paderborn")
        h = mdl(b_future, temp, b_past, h_past)
        assert h.shape == (1, future), f"h has shape {h.shape} for material {m_lbl}"
        assert np.isfinite(h).all(), f"non-finite H for material {m_lbl}"
