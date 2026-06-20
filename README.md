
<div align="center">
<h1>MagNet Toolkit</h1> 
<h2>Certified Models of the MagNet Challenge</h2>
</div>

![Lint and Test](https://github.com/upb-lea/mag-net-hub/actions/workflows/python-package.yml/badge.svg)

This repository provides a unified interface for certified magnetic component models from both the [MagNet Challenge 2023](https://github.com/minjiechen/magnetchallenge) and the [MagNet Challenge 2025](https://github.com/minjiechen/magnetchallenge-2).
It hosts two families of models under a common API:

- **Loss models** accept a periodic $B$ waveform, frequency, temperature, and material to produce a scalar power loss estimate together with a corresponding periodic $H$ waveform (optional).
- **Sequence-to-sequence models** predict a future $H$ sequence directly from a short warmup window of past $B$ / $H$ samples, a future $B$ sequence, and the temperature — operating on variable-length waveforms without resampling.

Feel free to use these models for your power converter design as a complement to your datasheet.

__Disclaimer__: Only steady-state and no varying DC-Bias is supported yet. 
Moreover, training data stemmed from measurements on toroid-shaped ferrites that had a fix size.

Supported materials for the loss models:
- 3C90, 3C92, 3C94, 3C95, 3E6, 3F4, 77, 78, 79, ML95S, N27, N30, N49, N87, T37

Supported materials for the sequence-to-sequence models:
- 3C90, 3C92, 3C94, 3C95, 3E6, 3F4, 77, 78, FEC007, FEC014, N27, N30, N49, N87, T37


## Installation

### Python
We strongly recommend Python __3.13__.
Higher versions may also work.

Then install through pip:

```
pip install mag-net-hub
```

or, alternatively, clone this repo and execute

```
cd mag-net-hub
pip install .
```

If you use `uv`, then just add to dependencies:
```
uv add mag-net-hub
```

## Usage
Models are provided as executable code with readily trained coefficients.
Hence, no training is conducted in this project.

### Python

#### Power loss models
These models assume periodic B field signals. Any given sequence is resampled to a model-specific sample rate automatically.

```py
import numpy as np
import magnethub as mh

# instantiate material-specific model
mdl = mh.loss.LossModel(material="3C92", team="paderborn")

# dummy B field data (one trajectory with 1024 samples)
b_wave = np.random.randn(1024)* 200e-3  # in T
freq = 124062  # Hz
temp = 58  # °C

# get power loss in W/m³ and estimated H wave in A/m
p, h = mdl(b_wave, freq, temp)

# batch execution for 100 trajectories
b_waves = np.random.randn(100, 1024)* 200e-3  # in T
freqs = np.random.randint(100e3, 750e3, size=100)
temps = np.random.randint(20, 80, size=100)
p, h = mdl(b_waves, freqs, temps)

```

#### Sequence-to-sequence models

Sequence models predict a future $H$ waveform from a short warmup window of past $B$/$H$
samples plus a future $B$ waveform and the temperature. Any sequence length is accepted but a fix sample rate of 16 MHz is assumed.

```py
import numpy as np
import magnethub as mh

# instantiate material-specific sequence model
mdl = mh.sequence.SequenceModel(material="3C90", team="paderborn")

future = 1024
b_future = np.random.randn(future) * 200e-3  # future B window in T
temp = 25  # °C

# simplest call — no warmup, past defaults to zero
h = mdl(b_future, temp)  # H field in A/m

# with an explicit warmup window for better accuracy
warmup = 128
b_past = np.random.randn(warmup) * 200e-3   # past B warmup window in T
h_past = np.random.randn(warmup) * 5        # past H warmup window in A/m
h = mdl(b_future, temp, b_past, h_past)     # future H window in A/m

# batch execution for 100 trajectories
b_past = np.random.randn(100, warmup) * 200e-3
h_past = np.random.randn(100, warmup) * 5
b_future = np.random.randn(100, future) * 200e-3
temps = np.random.randint(20, 80, size=100)
h = mdl(b_future, temps, b_past, h_past)

```


## Contributing
Whether you want to contribute your submission to the MagNet Challenge, or you are a single contributor who wants to add an awesome model to this hub -- any contribution is welcome.

Open a pull request to directly suggest small improvements to the infrastructure or to add your model (with performance statistics preferred). 
For larger suggestions, please first open an issue or go to the discussion section to discuss your ideas. 

See the below folder structure overview with annotations on how to contribute a model.

```
.
├── src_py
│   └── magnethub
│       ├── __init__.py
│       ├── loss.py
│       ├── sequence.py
│       ├── models
│       │   ├── paderborn
│       │   │   ├── changelog.md
│       │   │   ├── cnn_3C90_experiment_1b4d8_model_f3915868_seed_0_fold_0.pt
│       │   │   ├── cnn_3C92_experiment_ea1fe_model_72510647_seed_0_fold_0.pt
|       |   |   └──  ...
│       │   ├── paderborn_sequence_modeling
│       │   │   ├── 3C90_GRU8_MagNetHub-reduced-features-f32_5c3f8051_seed49.eqx
│       │   │   ├── example_usage.py
│       │   │   └──  ...
│       │   ├── sydney
│       │   │   └── ...
│       │   └── <add your contributor folder here>
│       │   │   └── <add your model coefficients here>
│       ├── paderborn_loss.py
│       ├── paderborn_sequence.py
|       ├── sydney.py
|       ├── <add your model code here>

```

Any number of models can be incorporated easily according to this code structure policy.
If you have added model coefficients and execution logic via code, it only requires to be hooked in
`loss.py` (or `sequence.py` for sequence-to-sequence models) and you are ready to fire this pull request (PR).

If it is possible, please also consider adding tests for your model logic under `tests/`, writing comprehensive docstrings in your code with some comments, and discuss the performance of your model in your PR. 

Other open ToDos are:
 - Matlab implementation
 - improve documentation

Thank you!
