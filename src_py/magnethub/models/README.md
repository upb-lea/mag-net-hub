# Model coefficients

This folder holds the trained coefficients of all models in the MagNet Toolkit.
The model code lives one level up (`magnethub/*.py`); the files here are loaded by it and are not meant to be used directly.

| Folder | Team | Model type | Public API | Code |
|---|---|---|---|---|
| `paderborn/` | Paderborn University (MagNet Challenge 2023) | CNN power loss model | `mh.loss.LossModel(team="paderborn")` | `paderborn_loss.py` |
| `sydney/` | University of Sydney (MagNet Challenge 2023) | Power loss model | `mh.loss.LossModel(team="sydney")` | `sydney.py` |
| `siegen_paderborn_sequence_modeling/` | University of Siegen & Paderborn University (MagNet Challenge 2025) | GRU sequence-to-sequence H estimator | `mh.sequence.SequenceModel(team="siegen_paderborn")` | `siegen_paderborn_sequence.py` |

## What the models are for

**Loss models** (MagNet Challenge 2023) take one period of a steady-state $B$ waveform, its frequency and the temperature, and estimate the volumetric power loss and, optionally, the periodic $H$ waveform.
The input is resampled to the model-specific sequence length internally.

**Sequence-to-sequence models** (MagNet Challenge 2025) take a short warmup window of measured past $B$ and $H$ samples, a future $B$ sequence of arbitrary length and the temperature, and predict the future $H$ sequence.
They are recurrent models (GRU with 8 hidden states) that run sample by sample at a fixed rate of **16 MHz**; the input is *not* resampled.
The models were trained on the MagNet Challenge 2025 data, which consists of measured, mostly switched (piecewise-linear) excitation.
They are adapted from [RHINO-MAG](https://github.com/upb-lea/RHINO-MAG).

## Validity range of the sequence models

The sequence models are purely data-driven and **do not extrapolate reliably**.
Outside the range of their training data they can produce unphysical results, e.g., an $H$ loop with a large DC offset under symmetric excitation, wrong amplitudes, or a prediction that strongly depends on the warmup window (see [issue #11](https://github.com/upb-lea/mag-net-hub/issues/11)).

The most relevant constraint is the **slope** $|\mathrm{d}B/\mathrm{d}t|$.
The models use the per-sample first and second gradient of $B$ as input features.
These are normalized by their maximum in the training set, so any slope beyond that maximum has never been seen during training.
Large flux density amplitudes at high frequencies quickly violate this limit, e.g., a 0.4 T triangle at 100 kHz for 3C90.

`SequenceModel` checks every call and emits an `OutOfTrainingRangeWarning` if any of the following exceeds its training maximum:
- $|B|$ (column *max. B* below),
- $|\mathrm{d}B/\mathrm{d}t|$ (column *max. dB/dt* below),
- $|\mathrm{d}^2B/\mathrm{d}t^2|$, which mostly matters for the sharp corners of ideal triangular waveforms,
- the temperature $T$ (70 °C for all materials).

For a symmetric triangle with peak $\hat B$ and frequency $f$, the slope is $4 \hat B f$; for a sine it is $2\pi \hat B f$.
The last two columns give the resulting limits on $\hat B \cdot f$ (peak flux density times frequency):

| Material | max. B in T | max. dB/dt in kT/s | Triangle: max. B·f in T·kHz | Sine: max. B·f in T·kHz |
|---|---|---|---|---|
| 3C90 | 0.508 | 147 | 37 | 23 |
| 3C92 | 0.469 | 79 | 20 | 13 |
| 3C94 | 0.528 | 85 | 21 | 13 |
| 3C95 | 0.534 | 93 | 23 | 15 |
| 3E6 | 0.462 | 87 | 22 | 14 |
| 3F4 | 0.349 | 104 | 26 | 17 |
| 77 | 0.491 | 84 | 21 | 13 |
| 78 | 0.517 | 85 | 21 | 13 |
| FEC007 | 0.483 | 94 | 23 | 15 |
| FEC014 | 0.451 | 122 | 30 | 19 |
| N27 | 0.504 | 128 | 32 | 20 |
| N30 | 0.396 | 129 | 32 | 21 |
| N49 | 0.505 | 130 | 33 | 21 |
| N87 | 0.453 | 136 | 34 | 22 |
| T37 | 0.387 | 123 | 31 | 20 |

Example: for 3C90, a triangle of 0.2 T peak is within the slope limit up to about 184 kHz, a 0.4 T triangle only up to about 92 kHz.

Staying below these limits is necessary but not sufficient.
The training data rarely combines a large amplitude with a steep slope, so predictions close to the limits, especially at high amplitude, should be treated with caution even when no warning is raised.

Further recommendations:
- Provide a measured (or at least physically plausible, i.e., hysteretic) $H$ warmup window. A synthetic warmup such as $H = k \cdot B$ is not hysteretic and can push the model into a wrong state.
- Sample the input at 16 MHz; the slope features are computed per sample.
- The warning can be silenced with `warnings.filterwarnings("ignore", category=magnethub.siegen_paderborn_sequence.OutOfTrainingRangeWarning)`.
