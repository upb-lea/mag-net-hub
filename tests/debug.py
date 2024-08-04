import numpy as np
import pandas as pd
from pathlib import Path
from magnethub.loss import LossModel, MATERIALS

test_ds = pd.read_csv(
    Path.cwd() / "tests" / "test_files" / "all_data.csv.gzip", dtype={"material": str}
)
errs_d = {}
for m_lbl in MATERIALS:
    mdl = LossModel(material=m_lbl, team="paderborn")
    test_mat_df = test_ds.query("material == @m_lbl")
    p, h = mdl(
        test_mat_df.loc[:, [c for c in test_mat_df if c.startswith("B_t_")]].to_numpy(),
        test_mat_df.loc[:, "freq"].to_numpy(),
        test_mat_df.loc[:, "temp"].to_numpy(),
    )
    rel_err = np.abs(test_mat_df.ploss - p) / test_mat_df.ploss
    errs_d[m_lbl] = {
        "avg": np.mean(rel_err),
        "95th": np.quantile(rel_err, 0.95),
        "99th": np.quantile(rel_err, 0.99),
        'samples': len(rel_err),
    }
rel_df = pd.DataFrame(errs_d).T
print(f"Rel. errors")
print(rel_df)
