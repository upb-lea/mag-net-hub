"""Loss model."""
from pathlib import Path
import mag_net_hub.paderborn as pb


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
    "paderborn": pb.MODEL_PATHS,
    "sydney": {},
}


class LossModel:
    """LossModel definition."""

    def __init__(self, material="3C92", team="paderborn"):
        self.material = material.upper()
        self.team = team.lower()

        # value checks
        if self.material not in MATERIALS:
            raise ValueError(
                f"Chosen material '{self.material}' not supported. Must be either {', '.join(MATERIALS)}"
            )
        if self.team not in list(TEAMS.keys()):
            raise ValueError(
                f"Chosen team '{self.team}' not supported. Must be in {', '.join(TEAMS.keys())}"
            )

        model_file_name = TEAMS[self.team].get(self.material, None)
        if model_file_name is None:
            raise ValueError(
                f"Team {self.team.capitalize()} does not offer a model for material {self.material}"
            )
        model_path = MODEL_ROOT / self.team / model_file_name

        # load corresponding model
        match self.team:
            case "paderborn":
                self.mdl = pb.PaderbornModel(model_path, self.material)
            case "sydney":
                raise NotImplementedError("Sydney model not implemented yet")

    def __call__(self, b_field, frequency, temperature):
        """Evaluate trajectory and estimate power loss.
        
        Args
        ----
        b_field: (B, T) array_like
            The magnetic flux density array(s). First dimension describes the batch, the second
             the time length (will always be interpolated to 1024 samples)
        frequency: scalar or 1D array-like
            The frequency operation point(s)
        temperature: scalar or 1D array-like
            The temperature operation point(s)
        """
        return self.mdl(b_field, frequency, temperature)
