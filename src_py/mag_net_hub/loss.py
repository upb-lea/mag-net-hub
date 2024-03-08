from pathlib import Path

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
    "paderborn": {
        "3C92": "cnn_A_experiment_c9cfe_model_d893c778_seed_0_fold_0.pt",
        "T37": "cnn_B_experiment_c9cfe_model_b6a920cc_seed_0_fold_0.pt",
        "3C95": "cnn_C_experiment_c9cfe_model_c1ced7b6_seed_0_fold_0.pt",
        "79": "cnn_D_experiment_c9cfe_model_11672810_seed_0_fold_0.pt",
        "ML95S": "cnn_E_experiment_c9cfe_model_5ae50f9e_seed_0_fold_0.pt",
    },
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
                f"Team {self.team} does not offer a model for material {self.material}"
            )
        model_path = MODEL_ROOT / self.team / model_file_name

        # TODO load corresponding model
        match self.team:
            case "paderborn":
                raise NotImplementedError()
            case "sydney":
                raise NotImplementedError()

    def __call__(self, b_field, frequency, temperature, return_h_sequence=True):
        raise NotImplementedError()
