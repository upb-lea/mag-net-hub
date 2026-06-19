"""The general sequence model.

The SequenceModel class wraps sequence-to-sequence models from the MagNet Challenge 2025.
Unlike the first-generation LossModel that estimates power losses from B, frequency, and temperature,
SequenceModel takes a past B/H warmup window, a future B field sequence, and temperature, then outputs
a future H field sequence of the same length as the future B field.
"""

from pathlib import Path
import numpy as np

import magnethub.paderborn_sequence as pbs

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

# Teams are registered here as they provide models.
# Format: {"team_name": team_module.MAT2FILENAME}
TEAMS = {
    "paderborn": pbs.MAT2FILENAME,
}

# Sub-directory under ``MODEL_ROOT`` that holds each team's coefficient files.
TEAM_SUBDIR = {
    "paderborn": pbs.MODEL_SUBDIR,
}


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
        model_path = MODEL_ROOT / TEAM_SUBDIR[self.team] / model_file_name

        # Load the corresponding model—dispatch based on team.
        # New teams are added here as match cases when their backends land.
        match self.team:
            case "paderborn":
                self.mdl = pbs.PaderbornSequenceModel(model_path, self.material)
            case _:
                raise NotImplementedError(f"No model backend available yet for team '{self.team}'")

        # The loaded backend must expose:
        #   self.mdl(b_past, h_past, b_future, temperature) -> h_future   (np.ndarray)

    def __call__(self, b_future, temperature, b_past=None, h_past=None):
        """Estimate a future H field sequence from a warmup window and a future B sequence.

        Args
        ----
        b_future : array_like, shape (F,) or (X, F)
            Future magnetic flux density sequence in T for which H is predicted.  Any
            sequence length F is accepted.
        temperature : scalar or 1-D array-like
            Temperature operation point(s) in °C.
        b_past : array_like, shape (P,) or (X, P), optional
            Past magnetic flux density warmup window in T.  X is the batch size, P the
            number of warmup samples.  Any warmup length P is accepted.
            Defaults to a single zero sample when omitted.
        h_past : array_like, shape (P,) or (X, P), optional
            Past magnetic field strength warmup window in A/m, aligned with ``b_past``.
            Defaults to a single zero sample when omitted.

        Returns
        -------
        h_future : np.ndarray, shape (X, F)
            Estimated future magnetic field strength in A/m, same shape as the
            (possibly reshaped) input ``b_future``.
        """
        b_future = np.asarray(b_future, dtype=np.float64)
        if b_future.ndim == 1:
            b_future = b_future.reshape(1, -1)

        batch = b_future.shape[0]

        if b_past is None:
            b_past = np.zeros((batch, 1), dtype=np.float64)
        else:
            b_past = np.asarray(b_past, dtype=np.float64)
            if b_past.ndim == 0:
                b_past = b_past.reshape(1, 1).repeat(batch, axis=0)
            elif b_past.ndim == 1:
                b_past = b_past.reshape(1, -1)

        if h_past is None:
            h_past = np.zeros((batch, 1), dtype=np.float64)
        else:
            h_past = np.asarray(h_past, dtype=np.float64)
            if h_past.ndim == 0:
                h_past = h_past.reshape(1, 1).repeat(batch, axis=0)
            elif h_past.ndim == 1:
                h_past = h_past.reshape(1, -1)

        temperature = np.atleast_1d(np.asarray(temperature, dtype=np.float64))

        assert b_past.shape[0] == h_past.shape[0] == b_future.shape[0] == temperature.shape[0], (
            f"Batch sizes disagree: b_past={b_past.shape[0]}, h_past={h_past.shape[0]}, "
            f"b_future={b_future.shape[0]}, temperature={temperature.shape[0]}"
        )
        assert b_past.shape[1] == h_past.shape[1], (
            f"Warmup lengths disagree: b_past has {b_past.shape[1]} samples, h_past has {h_past.shape[1]}"
        )

        h_future = self.mdl(b_past, h_past, b_future, temperature)

        assert h_future.ndim == 2, (
            f"H sequence has ndim={h_future.ndim}, but 2 were expected with (#sequences, #samples-per-sequence)"
        )
        assert h_future.shape == b_future.shape, (
            f"H sequence shape {h_future.shape} does not match future B field shape {b_future.shape}"
        )

        return h_future
