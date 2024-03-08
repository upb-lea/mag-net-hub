from pathlib import Path

MATERIALS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "3C90",
    "3C94",
    "3E6",
    "3F4",
    "77",
    "78",
    "N27",
    "N30",
    "N49",
    "N87",
]

MODEL_ROOT = Path(__file__).parent / "models"

TEAMS = {
    "paderborn": {
        "A": "cnn_A_experiment_c9cfe_model_d893c778_seed_0_fold_0.pt",
        "B": "cnn_B_experiment_c9cfe_model_b6a920cc_seed_0_fold_0.pt",
        "C": "cnn_C_experiment_c9cfe_model_c1ced7b6_seed_0_fold_0.pt",
        "D": "cnn_D_experiment_c9cfe_model_11672810_seed_0_fold_0.pt",
        "E": "cnn_E_experiment_c9cfe_model_5ae50f9e_seed_0_fold_0.pt",
    },
    "sydney": {},
}


class LossModel:

    def __init__(self, material="A", team="paderborn", return_h_sequence=True):
        self.material = material.upper()
        self.team = team.lower()
        self.return_h_sequence = return_h_sequence

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

    def __call__(self, b_field, frequency, temperature):
        raise NotImplementedError()
