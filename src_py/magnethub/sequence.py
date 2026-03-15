"""The general sequence model.

The SequenceModel class wraps sequence-to-sequence models from the MagNet Challenge 2025.
Unlike the first-generation LossModel that estimates power losses from B, frequency, and temperature,
SequenceModel takes a B field sequence, an initial H field guess, and temperature, then outputs
an H field sequence of the same length as the input B field.
"""

from pathlib import Path
import numpy as np

MATERIALS = {
    "T37",
    "3C90",
    "3C92",
    "3C94",
    "3C95",
    "3E6",
    "3F4",
    "77",
    "78",
    "FEC007",
    "FEC014",
    "N27",
    "N30",
    "N49",
    "N87",
}

MODEL_ROOT = Path(__file__).parent / "models"

# Teams will be registered here as they provide models.
# Format: {"team_name": team_module.MAT2FILENAME}
TEAMS = {}


class SequenceModel:
    """SequenceModel definition for MagNet Challenge 2025 models.

    Wraps sequence-to-sequence H field estimators that operate directly on
    variable-length B field waveforms (no resampling to a fixed length).
    """

    def __init__(self, material="3C92", team="paderborn"):
        self.material = material.upper()
        self.team = team.lower()

        if self.material not in MATERIALS:
            raise ValueError(f"Chosen material '{self.material}' not supported. Must be either {', '.join(MATERIALS)}")
        if self.team not in TEAMS:
            raise ValueError(
                f"Chosen team '{self.team}' not supported. Must be in {', '.join(TEAMS.keys()) if TEAMS else '(none registered yet)'}"
            )

        model_file_name = TEAMS[self.team].get(self.material, None)
        if model_file_name is None:
            raise ValueError(f"Team {self.team.capitalize()} does not offer a model for material {self.material}")
        model_path = MODEL_ROOT / self.team / model_file_name

        # Load the corresponding model—dispatch based on team.
        # New teams are added here as match cases when their backends land.
        match self.team:
            case _:
                raise NotImplementedError(f"No model backend available yet for team '{self.team}'")

        # After a real backend is loaded above, self.mdl must expose:
        #   self.mdl(b_field, h_initial, temperature) -> h_seq   (np.ndarray)

    def __call__(self, b_field, h_initial, temperature):
        """Evaluate B field sequence and estimate H field sequence.

        Args
        ----
        b_field : array_like, shape (Y,) or (X, Y)
            Magnetic flux density waveform(s) in T.  X is the batch size, Y the
            number of time-samples per period.  Any sequence length Y is accepted.
        h_initial : scalar or 1-D array-like
            Initial H field guess(es) in A/m from which the sequence integration
            starts.  Scalar is broadcast to the batch; 1-D must match batch size X.
        temperature : scalar or 1-D array-like
            Temperature operation point(s) in °C.

        Returns
        -------
        h : np.ndarray, shape (X, Y)
            Estimated magnetic field strength in A/m, same shape as the
            (possibly reshaped) input ``b_field``.
        """
        b_field = np.asarray(b_field, dtype=np.float64)
        if b_field.ndim == 1:
            b_field = b_field.reshape(1, -1)

        h_initial = np.atleast_1d(np.asarray(h_initial, dtype=np.float64))
        temperature = np.atleast_1d(np.asarray(temperature, dtype=np.float64))

        h_seq = self.mdl(b_field, h_initial, temperature)

        assert h_seq.ndim == 2, (
            f"H sequence has ndim={h_seq.ndim}, but 2 were expected with (#sequences, #samples-per-sequence)"
        )
        assert h_seq.shape == b_field.shape, (
            f"H sequence shape {h_seq.shape} does not match B field shape {b_field.shape}"
        )

        return h_seq
