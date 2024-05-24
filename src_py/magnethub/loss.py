"""The general loss model.

The LossModel class wraps all other teams' models.
It sanitizes user arguments that would be boilerplate code for any team's code.
"""

from pathlib import Path
import magnethub.paderborn as pb
import magnethub.sydney as sy
import numpy as np

L = 1024  # expected sequence length


MATERIALS = [
    "ML95S",
    "T37",
    "3C90",
    "3C92",
    "3C94",
    "3C95",
    "3E6",
    "3F4",
    "77",
    "78",
    "79",
    "N27",
    "N30",
    "N49",
    "N87",
]

MODEL_ROOT = Path(__file__).parent / "models"

TEAMS = {
    "paderborn": pb.MAT2FILENAME,
    "sydney": sy.MAT2FILENAME,
}


class LossModel:
    """LossModel definition."""

    def __init__(self, material="3C92", team="paderborn"):
        self.material = material.upper()
        self.team = team.lower()

        # value checks
        if self.material not in MATERIALS:
            raise ValueError(f"Chosen material '{self.material}' not supported. Must be either {', '.join(MATERIALS)}")
        if self.team not in list(TEAMS.keys()):
            raise ValueError(f"Chosen team '{self.team}' not supported. Must be in {', '.join(TEAMS.keys())}")

        model_file_name = TEAMS[self.team].get(self.material, None)
        if model_file_name is None:
            raise ValueError(f"Team {self.team.capitalize()} does not offer a model for material {self.material}")
        model_path = MODEL_ROOT / self.team / model_file_name

        # load corresponding model
        match self.team:
            case "paderborn":
                self.mdl = pb.PaderbornModel(model_path, self.material)
            case "sydney":
                self.mdl = sy.SydneyModel(model_path, self.material)

    def __call__(self, b_field, frequency, temperature):
        """Evaluate trajectory and estimate power loss.

        Args
        ----
        b_field: (X, Y) array_like
            The magnetic flux density array(s) in T. First dimension X describes the batch size, the second Y
             the time length (will always be interpolated to 1024 samples)
        frequency: scalar or 1D array-like
            The frequency operation point(s) in Hz
        temperature: scalar or 1D array-like
            The temperature operation point(s) in °C

        Return
        ------
        p, h: (X,) np.array, (X, Y) np.ndarray
            The estimated power loss (p) in W/m³ and the estimated magnetic field strength (h) in A/m.
        """
        if b_field.ndim == 1:
            b_field = b_field.reshape(1, -1)
        original_seq_len = b_field.shape[-1]

        L = self.mdl.expected_seq_len
        if b_field.shape[-1] != L:
            actual_len = b_field.shape[-1]
            query_points = np.arange(L)
            support_points = np.arange(actual_len) * L / actual_len
            # TODO Does a vectorized form of 1d interpolation exist?
            b_field = np.row_stack(
                [np.interp(query_points, support_points, b_field[i]) for i in range(b_field.shape[0])]
            )

        p, h_seq = self.mdl(b_field, frequency, temperature)

        if h_seq is not None:
            assert (
                h_seq.ndim == 2
            ), f"H sequence has ndim={h_seq.ndim}, but 2 were expected with (#periods, #samples-per-period)"
            # may interpolate to original sample size if h_seq too short or too long
            if h_seq.shape[-1] != original_seq_len:
                actual_len = h_seq.shape[-1]
                query_points = np.arange(original_seq_len)
                support_points = np.arange(actual_len) * original_seq_len / actual_len
                h_seq = np.row_stack([np.interp(query_points, support_points, h_seq[i]) for i in range(h_seq.shape[0])])
        return p, h_seq
