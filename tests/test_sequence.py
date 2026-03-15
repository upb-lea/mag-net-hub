import numpy as np
import pytest
from magnethub.sequence import SequenceModel, MATERIALS, TEAMS


class TestSequenceModelValidation:
    """Tests for input validation and error handling."""

    def test_invalid_material_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            SequenceModel(material="INVALID", team="paderborn")

    def test_invalid_team_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            SequenceModel(material="3C92", team="nonexistent")

    def test_no_teams_registered(self):
        """With no teams registered, any team name should fail."""
        assert len(TEAMS) == 0, "Expected no teams to be registered yet"
        with pytest.raises(ValueError, match="not supported"):
            SequenceModel(material="3C92", team="paderborn")

    def test_case_insensitive_material(self):
        """Material should be uppercased internally, but still fail if no team is available."""
        with pytest.raises(ValueError, match="not supported"):
            SequenceModel(material="3c92", team="someteam")

    def test_case_insensitive_team(self):
        """Team should be lowercased internally."""
        with pytest.raises(ValueError, match="not supported"):
            SequenceModel(material="3C92", team="SOMETEAM")


def _fake_backend(b, h0, t):
    """Stub backend that returns zeros with the same shape as the B input."""
    return np.zeros_like(b)


class TestSequenceModelCall:
    """Tests for the __call__ interface using a stubbed model backend."""

    @pytest.fixture()
    def model(self):
        """Create a SequenceModel with a fake inner model, bypassing __init__."""
        mdl = object.__new__(SequenceModel)
        mdl.material = "3C92"
        mdl.team = "faketeam"
        mdl.mdl = _fake_backend
        return mdl

    def test_single_waveform(self, model):
        b_wave = np.random.randn(512) * 200e-3
        h_initial = 0.0
        temp = 50.0

        h = model(b_wave, h_initial, temp)

        assert h.shape == (1, 512)
        assert h.dtype == np.float64

    def test_batch_waveform(self, model):
        b_waves = np.random.randn(10, 256) * 200e-3
        h_initials = np.zeros(10)
        temps = np.full(10, 60.0)

        h = model(b_waves, h_initials, temps)

        assert h.shape == (10, 256)

    def test_variable_length_accepted(self, model):
        """Any sequence length should be accepted without resampling."""
        for length in [64, 100, 233, 1024, 2048]:
            b = np.random.randn(1, length) * 200e-3
            h = model(b, 0.0, 50.0)
            assert h.shape == (1, length), f"Failed for length {length}"

    def test_scalar_h_initial(self, model):
        b = np.random.randn(5, 128) * 200e-3
        h = model(b, 0.0, 50.0)
        assert h.shape == (5, 128)

    def test_array_h_initial(self, model):
        b = np.random.randn(3, 128) * 200e-3
        h_init = np.array([1.0, 2.0, 3.0])
        h = model(b, h_init, np.array([50.0, 60.0, 70.0]))
        assert h.shape == (3, 128)

    def test_inner_model_receives_correct_args(self, model):
        """Verify that __call__ reshapes 1-D input and wraps scalars."""
        captured = {}

        def spy(b, h0, t):
            captured["b"] = b
            captured["h0"] = h0
            captured["t"] = t
            return np.zeros_like(b)

        model.mdl = spy
        model(np.random.randn(200) * 200e-3, 5.0, 42.0)

        assert captured["b"].shape == (1, 200)
        np.testing.assert_array_equal(captured["h0"], np.array([5.0]))
        np.testing.assert_array_equal(captured["t"], np.array([42.0]))

    def test_output_shape_mismatch_raises(self, model):
        """If the backend returns wrong shape, the assertion should catch it."""
        model.mdl = lambda b, h0, t: np.zeros((b.shape[0], b.shape[1] + 1))
        b = np.random.randn(1, 128) * 200e-3
        with pytest.raises(AssertionError, match="does not match"):
            model(b, 0.0, 50.0)

    def test_output_ndim_mismatch_raises(self, model):
        """If the backend returns 1-D instead of 2-D, the assertion should catch it."""
        model.mdl = lambda b, h0, t: np.zeros(b.shape[1])
        b = np.random.randn(1, 128) * 200e-3
        with pytest.raises(AssertionError, match="ndim"):
            model(b, 0.0, 50.0)
