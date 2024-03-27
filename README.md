
<div align="center">
<h1>MagNet Toolkit</h1> 
<h2>Certified Models of the MagNet Challenge</h2>
</div>

![Lint and Test](https://github.com/upb-lea/mag-net-hub/actions/workflows/python-package.yml/badge.svg)

This repository acts as a hub for selected power loss models that were elaborated by different competitors during the [MagNet Challenge 2023](https://github.com/minjiechen/magnetchallenge).
Feel free to use these loss models for your power converter design as a complement to your datasheet.

Note that they only support steady-state and no varying DC-Bias yet.

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
```
pip install mag_net_hub
```

### Matlab
TBD


## Usage
Models are provided as executable code with readily trained coefficients.
Hence, no training is conducted in this project.

### Python
```py
import numpy as np
import mag_net_hub as mnh

# instantiate material-specific model
mdl = mnh.loss.LossModel(material="3C92", team="paderborn")

# dummy B field data (one trajectory with 1024 samples)
b_wave = np.random.randn(1024)* 200e-3  # mT
freq = 124062  # Hz
temp = 58  # °C

# get power loss and estimated H wave
p, h = mdl(b_wave, freq, temp)

# batch execution for 100 trajectories
b_waves = np.random.randn(100, 1024)* 200e-3  # mT
freqs = np.random.randint(100e3, 750e3, size=100)
temps = np.random.randint(20, 80, size=100)
p, h = mdl(b_waves, freqs, temps)

```

### Matlab
TBD


## Contributing
Open a pull request to directly suggest small improvements. For larger suggestions, first open an issue to discuss your ideas. 