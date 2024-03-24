import numpy as np
import pandas as pd
from pathlib import Path
import pytest
from mag_net_hub.loss import LossModel, MATERIALS


def test_smoke():
    mdl = LossModel(material="3C92", team="paderborn")
    # dummy B field data (one trajectory with 1024 samples)
    b_wave = np.random.randn(1024) * 200e-3  # mT
    freq = 124062  # Hz
    temp = 58  # °C

    # get loss and estimated H wave
    p, h = mdl(b_wave, freq, temp)
    assert np.isscalar(p), f"p has shape {p.shape}"
    assert h.shape == (1, 1024), f"h has shape {h.shape}"
    
    # repetition test
    p2, h2 = mdl(b_wave, freq, temp)
    assert np.allclose(p, p2), f"{p} != {p2}"
    assert np.allclose(h, h2), f"{h} != {h2}"
    


def test_shorter_sequence():
    mdl = LossModel(material="3C92", team="paderborn")
    # dummy B field data (one trajectory with 1024 samples)
    b_wave = np.random.randn(233) * 200e-3  # mT
    freq = 120_000  # Hz
    temp = 77  # °C

    # get scalar power loss
    p, h = mdl(b_wave, freq, temp)

    assert np.isscalar(p), f"p has shape {p.shape}"
    assert h.shape == (1, 1024), f"h has shape {h.shape}"


def test_batch_execution():
    mdl = LossModel(material="3C92", team="paderborn")

    b_waves = np.random.randn(100, 1024) * 200e-3  # mT
    freqs = np.random.randint(100e3, 750e3, size=100)
    temps = np.random.randint(20, 80, size=100)
    p, h = mdl(b_waves, freqs, temps)

    assert p.size == 100, f"{p.size=}"
    assert h.shape == (100, 1024), f"{h.shape=}"

def test_material_availability():
    b_wave = np.random.randn(1024) * 200e-3  # mT
    freq = 124062  # Hz
    temp = 58  # °C

    for m_lbl in MATERIALS:
        mdl = LossModel(material=m_lbl, team='paderborn')
        p, h = mdl(b_wave, freq, temp)
        assert np.isscalar(p), f"p has shape {p.shape}"
        assert h.shape == (1, 1024), f"h has shape {h.shape}"

@pytest.mark.skip(reason="DEBUG test. The test files do not exist except for the author.")
def test_full_accuracy():
    test_ds = pd.read_csv(Path.cwd() / 'tests' /'test_files'/ 'all_data.csv.gzip', dtype={'material': str})
    for m_lbl in MATERIALS:
        mdl = LossModel(material=m_lbl, team='paderborn')
        test_mat_df = test_ds.query("material == @m_lbl")
        p, h = mdl(test_mat_df.loc[:, [c for c in test_mat_df if c.startswith("B_t_")]].to_numpy(),
                   test_mat_df.loc[:, "freq"].to_numpy(),
                   test_mat_df.loc[:, 'temp'].to_numpy())
        rel_err = np.abs(test_mat_df.ploss - p)/ test_mat_df.ploss
        print(f"{m_lbl}: avg. rel err {np.mean(rel_err):.5%} 95th quantile {np.quantile(rel_err, 0.95):.5%}")
        #assert avg_rel_err < 0.70, f"Inaccurate for material {m_lbl} with prediction: {np.abs(test_mat_df.ploss - p)/ test_mat_df.ploss} W/m³"

@pytest.mark.skip(reason="DEBUG test. The test files do not exist except for the author.")
def test_full_accuracy_line_by_line():
    test_ds = pd.read_csv(Path.cwd() / 'tests' /'test_files'/ 'all_data.csv.gzip', dtype={'material': str})
    for m_lbl in MATERIALS:
        mdl = LossModel(material=m_lbl, team='paderborn')
        test_mat_df = test_ds.query("material == @m_lbl")
        p_preds_l = []
        
        for _, row in test_mat_df.iterrows():
            p, h = mdl(row.loc[[c for c in test_mat_df if c.startswith("B_t_")]].to_numpy().astype(np.float64),
                    row.loc["freq"],
                    row.loc['temp'])
            p_preds_l.append(p)
        p = np.array(p_preds_l)
        rel_err = np.abs(test_mat_df.ploss - p)/ test_mat_df.ploss
        print(f"{m_lbl}: avg. rel err {np.mean(rel_err):.5%} 95th quantile {np.quantile(rel_err, 0.95):.5%}")


def test_accuracy_slightly():
    test_ds = pd.read_csv(Path.cwd() / 'tests' /'test_files'/ 'unit_test_data_ploss_at_450kWpm3.csv', dtype={'material': str})
    for m_lbl in MATERIALS:
        mdl = LossModel(material=m_lbl, team='paderborn')
        test_mat_df = test_ds.query("material == @m_lbl")
        p, h = mdl(test_mat_df.loc[:, [c for c in test_mat_df if c.startswith("B_t_")]].to_numpy(),
                   test_mat_df.loc[:, "freq"].to_numpy(),
                   test_mat_df.loc[:, 'temp'].to_numpy())
        avg_rel_err = np.mean(np.abs(test_mat_df.ploss - p)/ test_mat_df.ploss)
        assert avg_rel_err < 0.08, f"Inaccurate for material {m_lbl} with prediction: {np.abs(test_mat_df.ploss - p)/ test_mat_df.ploss} W/m³"