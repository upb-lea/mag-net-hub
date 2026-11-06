"""Kirchgässner (2026) sequence-to-sequence H-field estimators: the classical and the MLP16-scheduled equivalent circuit.

The B-H behavior of each material is an equivalent circuit built from classical elements of hysteresis modeling,
discretized at the 16 MHz sampling step: a measurement correction of B, a reversible saturation curve, six viscous
play operators (a Prandtl-Ishlinskii model), two linear relaxations and a mean-field feedback of H into every operator,
followed by a first-order lag of the measured H. The classical circuit has 32 parameters per material, all at most
linear in temperature. In the MLP16-scheduled circuit, a small MLP per material (4 -> 16 -> 16 -> 16) maps the
excitation at every step to shifts of the element values, while the hysteresis memory stays in the circuit states.

Both start from the measured warmup window with a learned state estimator (initial play states, observer gains and
amplitude factors of the window) and then run open loop over the future B sequence. The output is the field as
measured by the MagNet Challenge 2 test bench. This module is a NumPy (float64) port of the PyTorch models; it
reproduces them to floating-point rounding.

Source: https://github.com/wkirgsn/ferrite-bh-circuit
"""

import json
import warnings
from pathlib import Path

import numpy as np

from magnethub.siegen_paderborn_sequence import OutOfTrainingRangeWarning

MODEL_SUBDIR = "kirchgaessner2026"
MATERIALS = ["3C90", "3C92", "3C94", "3C95", "3E6", "3F4", "77", "78", "FEC007", "FEC014", "N27", "N30", "N49", "N87",
             "T37"]
MAT2FILENAME = {m: f"{m}_classical.json" for m in MATERIALS}
MAT2FILENAME_MLP16 = {m: f"{m}_mlp16.json" for m in MATERIALS}

# order of the 16 MLP outputs: shifts of the static part (5) and of the operator dynamics (11)
OUTS = ["a1", "a3", "bs", "bw", "m", "al", "r0", "r1", "d0", "d1", "be1", "be2", "c1", "c2", "g0", "g1"]


def _sigmoid(x):
    """Logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    """Softplus log(1 + e^x), overflow-safe."""
    return np.logaddexp(0.0, x)


def _detrend(x, tt):
    """Remove the mean and the linear trend (along tt) of x along the last axis."""
    y = x - x.mean(-1, keepdims=True)
    return y - (y * tt).sum(-1, keepdims=True) / (tt * tt).sum() * tt


class Kirchgaessner2026Circuit:
    """Equivalent circuit of one material: classical (constant element values) or MLP-scheduled.

    The coefficient file holds the circuit and estimator parameters as trained (``params``), the training range of the
    inputs (``validity``) and, for the scheduled circuit, the MLP weights (``mlp``).
    """

    def __init__(self, model_path, material):
        """Load the coefficients of ``material`` from the JSON file ``model_path``."""
        self.model_path = Path(model_path)
        self.material = material
        rec = json.loads(self.model_path.read_text())
        self.M, self.J = rec["M"], rec["J"]
        self.p = {k: np.asarray(v, dtype=np.float64) for k, v in rec["params"].items()}
        self.mlp = {k: np.asarray(v, dtype=np.float64) for k, v in rec["mlp"].items()} if rec.get("mlp") else None
        self.validity = rec.get("validity")

    def __call__(self, b_past, h_past, b_future, temperature):
        """Estimate the future H field sequence.

        Args
        ----
        b_past : (X, P) np.ndarray
            Past magnetic flux density warmup window in T.
        h_past : (X, P) np.ndarray
            Past magnetic field strength warmup window in A/m, aligned with ``b_past``.
        b_future : (X, F) np.ndarray
            Future magnetic flux density sequence in T for which H is predicted.
        temperature : (X,) np.ndarray
            Temperature operation point(s) in °C.

        Return
        ------
        h_future : (X, F) np.ndarray
            The estimated future magnetic field strength in A/m.
        """
        if self.validity is not None:
            self._check_input_range(b_past, b_future, temperature)
        k = b_past.shape[1]
        B = np.concatenate([b_past, b_future], axis=1)
        H = np.concatenate([h_past, np.zeros_like(b_future)], axis=1)
        return self.predict(B, H, np.broadcast_to(temperature, (B.shape[0],)), k)[:, k:]

    def shifts(self, u, d1, d2, t):
        """Per-step shifts of the element values (16, N, W) from the excitation, even in (u, dB, d2B)."""
        N, W = u.shape
        x = np.stack([u / 0.2, np.arcsinh(d1 / 1e-3), np.arcsinh(d2 / 1e-4), np.broadcast_to(t, (N, W))], -1)
        z = 0.0
        for sg in (1.0, -1.0):                                            # the input and its mirror image
            xs = x * np.array([sg, sg, sg, 1.0])
            a = np.tanh(xs @ self.mlp["w1"] + self.mlp["b1"])
            a = np.tanh(a @ self.mlp["w2"] + self.mlp["b2"])
            z = z + (a @ self.mlp["w3"] + self.mlp["b3"]) / 2
        return np.moveaxis(z, -1, 0)

    def predict(self, B, H, T, k):
        """H (N, W) for windows B (N, W) in T at 16 MHz with the measured H known for n < k, temperatures T (N,)."""
        P, M, J = self.p, self.M, self.J
        N, W = B.shape
        t = ((np.asarray(T, dtype=np.float64) - 50.0) / 25.0)[:, None]  # (N, 1)
        lin = lambda c: c[..., 0] + c[..., 1] * t                         # coefficient linear in t (no t^2 terms)

        hla = 0.6 * _sigmoid(P["hl"][0])                                  # lag pole of the H measurement
        HE = (H - hla * np.concatenate([H[:, :1], H[:, :-1]], 1)) / (1 - hla)   # de-lagged warmup
        d1 = np.gradient(B, axis=1)
        d2 = np.gradient(d1, axis=1)
        tau = lin(P["tau"])                                               # (N, 2)
        u = B + tau[:, :1] * d1 + tau[:, 1:] * d2                         # measurement correction
        if self.mlp is not None:
            z = dict(zip(OUTS, self.shifts(u, d1, d2, t), strict=True))               # (N, W) each
        else:
            z = {o: np.zeros((N, W)) for o in OUTS}

        a1 = lin(P["a1"]) * np.exp(z["a1"])
        a3 = 100 * (lin(P["a3"]) + z["a3"])
        Bs = np.maximum(lin(P["bs"]) + 0.05 * z["bs"], 0.15)
        bw = np.exp(P["log_bw"][0] + z["bw"])
        e = lambda x: np.exp(np.minimum(x, 15.0))
        hsat = 10.0 * (e((u - Bs) / bw) - e((-u - Bs) / bw))
        static = a1 * u + a3 * u ** 3 + hsat
        mmod = np.exp(z["m"])

        lr0 = P["log_r"][0] + (P["log_r"][-1] - P["log_r"][0]) * np.linspace(0, 1, M)
        lr = lr0 + P["rT"][0] * t                                         # (N, M) log play widths
        x = (lr - lr.mean(-1, keepdims=True)) / (lr.std(-1, ddof=1, keepdims=True) + 1e-6)
        dw = np.concatenate([lin(P["dw"][0]), lin(P["dw"][1])], -1)       # (N, 2) play density d0, d1
        cz0 = np.concatenate([lin(P["cz"][j]) for j in range(J)], -1)     # (N, J) relaxation weights (raw)
        gop = np.concatenate([P["gop"][0] * x, np.broadcast_to(P["gop"][M:], (N, J))], -1)
        gamma0 = 1e-3 * (lin(P["gam"])[:, :, None] + gop[:, None, :])     # (N, 1, M + J)
        al0 = P["alpha"][0] + P["aT"][0] * t

        zs = lambda o: z[o][..., None]                                    # (N, W, 1)
        alpha = _sigmoid(al0[:, :, None] + zs("al"))                      # (N, W, 1)
        r = np.exp(lr[:, None, :] + zs("r0") + zs("r1") * x[:, None, :])  # (N, W, M)
        w = -_softplus(-((dw[:, None, :1] + zs("d0")) + (dw[:, None, 1:] + zs("d1")) * x[:, None, :]))
        beta = 0.5 * _sigmoid(P["beta"] + np.concatenate([zs(f"be{j + 1}") for j in range(J)], -1))
        cz = -_softplus(cz0[:, None, :] + np.concatenate([zs(f"c{j + 1}") for j in range(J)], -1))
        xg = np.concatenate([x, np.zeros((N, J))], -1)[:, None, :]
        gamma = gamma0 + 1e-3 * (zs("g0") + zs("g1") * xg)                # (N, W, M + J)
        al_k = np.concatenate([np.broadcast_to(alpha, (N, W, M)), beta], -1)
        r_k = np.concatenate([r, np.zeros((N, W, J))], -1)
        w_k = np.concatenate([w, cz], -1)

        # warmup estimator: prior initial states, initial mismatch spread over all states, observer gains
        u0 = u[:, :1]
        p = u0 - r[:, 0] * np.tanh(P["g2"] * u0 / r[:, 0] + np.sign(u[:, min(5, W - 1):min(5, W - 1) + 1] - u0) * P["g"])
        zz = np.repeat(u0, J, 1)
        e0 = HE[:, :1] - static[:, :1] - (w[:, 0] * p).sum(-1, keepdims=True) - (cz[:, 0] * zz).sum(-1, keepdims=True)
        K0 = P["K0"]
        p = p + 1e-3 * K0[:M] * e0
        zz = zz + 1e-3 * K0[M:M + J] * e0
        c = P["cpr"] + K0[-1] * e0                                        # (N, 1)
        s0 = np.concatenate([p, zz], -1)
        Lp, Lc = 1e-3 * P["L"][:M + J], P["L"][-1]

        def run(Wn, st, mm, kk):
            """Recurrence over the first Wn samples; the observer corrects the states for n < kk."""
            s, cc, hp = s0.copy(), c.copy(), HE[:, :1].copy()
            out = np.empty((N, Wn))
            for j in range(Wn):
                xx = u[:, j:j + 1] + gamma[:, j] * hp - s
                s = s + al_k[:, j] * (np.maximum(xx - r_k[:, j], 0.0) - np.maximum(-xx - r_k[:, j], 0.0))
                h = st[:, j:j + 1] + mm[:, j:j + 1] * (w_k[:, j] * s).sum(-1, keepdims=True) + cc
                if j < kk:
                    er = HE[:, j:j + 1] - h
                    s = s + Lp * er
                    cc = cc + Lc * er
                    h = HE[:, j:j + 1]
                out[:, j] = h[:, 0]
                hp = h
            return out

        if k >= 50:                                                       # amplitude factors of the window
            Ho = run(k, static, mmod, 1)
            tt = np.linspace(-1, 1, k - 1)
            yd = _detrend(HE[:, 1:k] - Ho[:, 1:], tt)
            n0 = np.exp(P["lq"][0] + P["lqk"][0] * np.log(k / 300))
            Sh = (_detrend(HE[:, 1:k], tt) ** 2).mean(-1, keepdims=True)
            h1 = _detrend(static[:, 1:k] - hsat[:, 1:k], tt)
            h2 = _detrend(Ho[:, 1:] - static[:, 1:k] - c, tt)
            s11, s22, s12 = ((a * b).sum(-1, keepdims=True) for a, b in ((h1, h1), (h2, h2), (h1, h2)))
            s11, s22 = s11 + n0 * Sh + 1e-6, s22 + n0 * Sh + 1e-6
            b1, b2 = (yd * h1).sum(-1, keepdims=True), (yd * h2).sum(-1, keepdims=True)
            det = s11 * s22 - s12 ** 2
            q = np.clip((s22 * b1 - s12 * b2) / det, -0.5, 0.5)
            qh = np.clip((s11 * b2 - s12 * b1) / det, -0.5, 0.5)
            static, mmod = static + q * (static - hsat), mmod * (1 + qh)
        out = run(W, static, mmod, k)

        y = np.empty_like(out)  # first-order lag of the H measurement
        yp = H[:, 0]
        for j in range(W):
            yp = hla * yp + (1 - hla) * out[:, j]
            y[:, j] = yp
        return y

    def _check_input_range(self, b_past, b_future, temperature):
        """Warn if T, |B|, its per-sample slope or curvature leave the range of the material's training data."""
        v = self.validity
        b = np.concatenate([b_past, b_future], axis=1)
        violations = []
        t = np.asarray(temperature, dtype=np.float64)
        if t.min() < v["T_min"] or t.max() > v["T_max"]:
            violations.append(f"T from {t.min():g} to {t.max():g} °C outside the trained {v['T_min']:g} to {v['T_max']:g} °C")
        if np.abs(b).max() > v["B_max"]:
            violations.append(f"|B| up to {np.abs(b).max() / v['B_max']:.2f}x the training maximum of {v['B_max']:.3f} T")
        if b.shape[1] > 1:
            db = np.gradient(b, axis=1)
            if np.abs(db).max() > v["dB_max"]:
                violations.append(f"|dB/dt| up to {np.abs(db).max() / v['dB_max']:.2f}x the training maximum of "
                                  f"{v['dB_max'] * 16e6:.3g} T/s (at 16 MHz sampling)")
            if np.abs(np.gradient(db, axis=1)).max() > v["d2B_max"]:
                violations.append(f"|d2B/dt2| up to {np.abs(np.gradient(db, axis=1)).max() / v['d2B_max']:.2f}x the "
                                  "training maximum")
        if violations:
            warnings.warn(
                f"Input for material {self.material} is outside the training range of the "
                + ("scheduling MLP" if self.mlp is not None else "classical circuit") + ": "
                + "; ".join(violations)
                + ". The H prediction is an extrapolation.",
                OutOfTrainingRangeWarning,
                stacklevel=4,  # point at the caller of SequenceModel.__call__
            )
