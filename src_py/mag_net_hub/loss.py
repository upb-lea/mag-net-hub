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

    def __call__(self, b_field, frequency, temperature, return_h_sequence=True):
        p, h = self.mdl(b_field, frequency, temperature)
        if return_h_sequence:
            return p, h
        else:
            return p
