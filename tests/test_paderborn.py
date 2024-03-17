import numpy as np
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