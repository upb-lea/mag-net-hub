
<div align="center">
<h1>MagNet Toolkit</h1> 
<h2>Certified Models of the MagNet Challenge</h2>
</div>

![Lint and Test](https://github.com/upb-lea/mag-net-hub/actions/workflows/python-package.yml/badge.svg)

This repository acts as a hub for selected power loss models that were elaborated by different competitors during the [MagNet Challenge 2023](https://github.com/minjiechen/magnetchallenge).
Feel free to use these loss models for your power converter design as a complement to your datasheet.

The loss models are designed such that you can request a certain frequency, temperature, material and $B$ wave (sequence), in order to be provided with a scalar power loss estimate and a corresponding $H$ wave estimate.

__Disclaimer__: Only steady-state and no varying DC-Bias is supported yet. 
Moreover, training data stemmed from measurements on toroid-shaped ferrites that had a fix size.

Supported materials:
- ML95S
- T37
- 3C90
- 3C92
- 3C94
- 3C95
- 3E6
- 3F4
- 77
- 78
- 79
- N27
- N30
- N49
- N87


## Installation

### Python
We strongly recommend Python __3.10__.
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

### Matlab
TBD


## Usage
Models are provided as executable code with readily trained coefficients.
Hence, no training is conducted in this project.

### Python
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

### Matlab
TBD


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
│       ├── models
│       │   ├── paderborn
│       │   │   ├── changelog.md
│       │   │   ├── cnn_3C90_experiment_1b4d8_model_f3915868_seed_0_fold_0.pt
│       │   │   ├── cnn_3C92_experiment_ea1fe_model_72510647_seed_0_fold_0.pt
|       |   |   └──  ...
│       │   ├── sydney
│       │   │   └── ...
│       │   └── <add your contributor folder here>
│       │   │   └── <add your model coefficients here>
│       ├── paderborn.py
|       ├── sydney.py
|       ├── <add your model code here>

```

Any number of models can be incorporated easily according to this code structure policy.
If you have added model coefficients and execution logic via code, it only requires to be hooked in
`loss.py` and you are ready to fire this pull request (PR).

If it is possible, please also consider adding tests for your model logic under `tests/`, writing comprehensive docstrings in your code with some comments, and discuss the performance of your model in your PR. 
Thank you!