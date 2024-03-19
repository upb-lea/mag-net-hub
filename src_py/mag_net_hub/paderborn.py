"""Source:
https://github.com/upb-lea/hardcore-magnet-challenge
"""

import numpy as np
import pandas as pd
import torch


L = 1024  # expected sequence length
ALL_B_COLS = [f"B_t_{k}" for k in range(L)]
ALL_H_COLS = [f"H_t_{k}" for k in range(L)]
FREQ_SCALE = 150_000.0  # in Hz

# material constants
MAT_CONST_B_MAX = {
    "3C90": 0.282254066096809,
    "3C94": 0.281179823717941,
    "3E6": 0.199842551960829,
    "3F4": 0.312954906128548,
    "77": 0.315644038162322,
    "78": 0.3166405692215,
    "3C92": 0.319193837507623,
    "T37": 0.253934182092085,
    "3C95": 0.322678797082694,
    "79": 0.314715273611617,
    "ML95S": 0.330102949741973,
    "N27": 0.317335585296054,
    "N30": 0.201167700159802,
    "N49": 0.317828937072173,
    "N87": 0.280909134946228,
}  # in T
MAT_CONST_H_MAX = {
    "3C90": 84.7148502254261,
    "3C94": 64.8575649838852,
    "3E6": 74.1579701817075,
    "3F4": 150,
    "77": 86.5681744566843,
    "78": 87.5896894086919,
    "3C92": 150,
    "T37": 87.7490689795367,
    "3C95": 72.0625845199264,
    "79": 150,
    "ML95S": 150,
    "N27": 119.039616554254,
    "N30": 116.951204964406,
    "N49": 150,
    "N87": 100.674197407678,
}  # in A/m

# Normalization constants for misc. input features (absolute maximum over entire data set)
NORM_DENOM = {
    "3C90": {
        "b_peak2peak": 0.562775,
        "log_peak2peak": 3.854492,
        "mean_abs_dbdt": 0.001098,
        "log_mean_abs_dbdt": 10.097290,
        "sample_time": 0.000020,
    },
    "3C92": {
        "b_peak2peak": 0.638772,
        "log_peak2peak": 3.952986,
        "mean_abs_dbdt": 0.001246,
        "log_mean_abs_dbdt": 10.191507,
        "sample_time": 0.000020,
    },
    "3C94": {
        "b_peak2peak": 0.561696288415998,
        "log_peak2peak": 3.8527982020339397,
        "mean_abs_dbdt": 0.001094609016982872,
        "log_mean_abs_dbdt": 10.09547460054903,
        "sample_time": 2.002002002002002e-05,
    },
    "3C95": {
        "b_peak2peak": 0.643972929264389,
        "log_peak2peak": 3.951775170875342,
        "mean_abs_dbdt": 0.001255971974873893,
        "log_mean_abs_dbdt": 10.19012406875038,
        "sample_time": 2.000400080016003e-05,
    },
    "3E6": {
        "b_peak2peak": 0.39860703974419,
        "log_peak2peak": 3.9943679877190745,
        "mean_abs_dbdt": 0.0007754383944842351,
        "log_mean_abs_dbdt": 10.237231774124638,
        "sample_time": 2.002002002002002e-05,
    },
    "3F4": {
        "b_peak2peak": 0.62579,
        "log_peak2peak": 3.966479191224205,
        "mean_abs_dbdt": 0.0012214485516744327,
        "log_mean_abs_dbdt": 10.204883300750092,
        "sample_time": 2.002002002002002e-05,
    },
    "77": {
        "b_peak2peak": 0.62941,
        "log_peak2peak": 3.969122597468719,
        "mean_abs_dbdt": 0.001227556207233634,
        "log_mean_abs_dbdt": 10.207264569238735,
        "sample_time": 2.002002002002002e-05,
    },
    "78": {
        "b_peak2peak": 0.6315575499770669,
        "log_peak2peak": 3.9789526676175275,
        "mean_abs_dbdt": 0.001231771095979432,
        "log_mean_abs_dbdt": 10.217365718778147,
        "sample_time": 2.002002002002002e-05,
    },
    "79": {
        "b_peak2peak": 0.648230254830443,
        "log_peak2peak": 3.9557162612448136,
        "mean_abs_dbdt": 0.0012638156025685287,
        "log_mean_abs_dbdt": 10.194239618633777,
        "sample_time": 2.00160128102482e-05,
    },
    "T37": {
        "b_peak2peak": 0.509931827049285,
        "log_peak2peak": 3.9681646195793543,
        "mean_abs_dbdt": 0.0009933972666621286,
        "log_mean_abs_dbdt": 10.207037964928544,
        "sample_time": 2.00160128102482e-05,
    },
    "N27": {
        "b_peak2peak": 0.6318699999999999,
        "log_peak2peak": 3.96936864553962,
        "mean_abs_dbdt": 0.0012323949169110505,
        "log_mean_abs_dbdt": 10.207788696606462,
        "sample_time": 2.002002002002002e-05,
    },
    "N30": {
        "b_peak2peak": 0.40126,
        "log_peak2peak": 3.9675357153584048,
        "mean_abs_dbdt": 0.0007812218963831858,
        "log_mean_abs_dbdt": 10.20594094230823,
        "sample_time": 2.002002002002002e-05,
    },
    "N49": {
        "b_peak2peak": 0.6349207696798069,
        "log_peak2peak": 3.729801733358157,
        "mean_abs_dbdt": 0.0012362016813373098,
        "log_mean_abs_dbdt": 9.972437816965455,
        "sample_time": 2.002002002002002e-05,
    },
    "N87": {
        "b_peak2peak": 0.561429326791338,
        "log_peak2peak": 3.8619698511811014,
        "mean_abs_dbdt": 0.001094995270801449,
        "log_mean_abs_dbdt": 10.104824015905193,
        "sample_time": 2.002002002002002e-05,
    },
    "ML95S": {
        "b_peak2peak": 0.660470219971137,
        "log_peak2peak": 3.684324864126149,
        "mean_abs_dbdt": 0.0012853467115523114,
        "log_mean_abs_dbdt": 9.922533485496238,
        "sample_time": 1.997602876548142e-05,
    },
}

MODEL_PATHS = {
    "3C90": "cnn_3C90_experiment_1b4d8_model_f3915868_seed_0_fold_0.pt",
    "3C92": "cnn_3C92_experiment_ea1fe_model_72510647_seed_0_fold_0.pt",
    "3C94": "cnn_3C94_experiment_56441_model_55693612_seed_0_fold_0.pt",
    "3C95": "cnn_3C95_experiment_ad6ec_model_046383a5_seed_0_fold_0.pt",
    "3E6": "cnn_3E6_experiment_a7817_model_a1035e58_seed_0_fold_0.pt",
    "3F4": "cnn_3F4_experiment_c234d_model_2d43b97d_seed_0_fold_0.pt",
    "77": "cnn_77_experiment_268ae_model_5b7c92ed_seed_0_fold_0.pt",
    "78": "cnn_78_experiment_e5297_model_77fbd758_seed_0_fold_0.pt",
    "79": "cnn_79_experiment_45989_model_7227af72_seed_0_fold_0.pt",
    "T37": "cnn_T37_experiment_a084a_model_fb31325e_seed_0_fold_0.pt",
    "N27": "cnn_N27_experiment_8b7f0_model_20954a2a_seed_0_fold_0.pt",
    "N30": "cnn_N30_experiment_5a78c_model_6bd86623_seed_0_fold_0.pt",
    "N49": "cnn_N49_experiment_27442_model_d6234a32_seed_0_fold_0.pt",
    "N87": "cnn_N87_experiment_985cb_model_d51b0f7e_seed_0_fold_0.pt",
    "ML95S": "cnn_ML95S_experiment_1c978_model_883a5d2f_seed_0_fold_0.pt",
}


def form_factor(x):
    """
    definition:      kf = rms(x) / mean(abs(x))
    for ideal sine:  np.pi/(2*np.sqrt(2))
    """
    return np.sqrt(np.mean(x**2, axis=1)) / np.mean(np.abs(x), axis=1)


def crest_factor(x):
    """
    definition:      kc = rms(x) / max(x)
    for ideal sine:  np.sqrt(2)
    """
    return np.max(np.abs(x), axis=1) / np.sqrt(np.mean(x**2, axis=1))


def bool_filter_sine(b, rel_kf=0.01, rel_kc=0.01, rel_0_dev=0.1):
    """
    b: input flux density (nxm)-array with n m-dimensional flux density waveforms
    rel_kf: (allowed) relative deviation of the form factor for sine classification
    rel_kc: (allowed) relative deviation of the crest factor for sine classification
    rel_0_dev: (allowed) relative deviation of the first value from zero (normalized on the peak value)
    """
    kf_sine = np.pi / (2 * np.sqrt(2))
    kc_sine = np.sqrt(2)

    b_ff = form_factor(b)
    b_cf = crest_factor(b)
    b_max = np.max(b, axis=1)
    mask = np.all(
        np.column_stack(
            [
                b_ff < kf_sine * (1 + rel_kf),  # form factor based checking
                b_ff > kf_sine * (1 - rel_kf),  # form factor based checking
                b_cf < kc_sine * (1 + rel_kc),  # crest factor based checking
                b_cf > kc_sine * (1 - rel_kc),  # crest factor based checking
                b[:, 0] < b_max * rel_0_dev,  # starting value based checking
                b[:, 0] > -b_max * rel_0_dev,  # starting value based checking
            ]
        ),
        axis=1,
    )

    return mask


def bool_filter_triangular(b, rel_kf=0.005, rel_kc=0.005):
    kf_triangular = 2 / np.sqrt(3)
    kc_triangular = np.sqrt(3)

    b_ff = form_factor(b)
    b_cf = crest_factor(b)

    mask = np.all(
        np.column_stack(
            [
                b_ff < kf_triangular * (1 + rel_kf),
                b_ff > kf_triangular * (1 - rel_kf),
                b_cf < kc_triangular * (1 + rel_kc),
                b_cf > kc_triangular * (1 - rel_kc),
            ]
        ),
        axis=1,
    )

    return mask


def get_waveform_est(full_b):
    """From Till's tp-1.4.7.3.1 NB, return waveform class.
    Postprocessing from wk-1.1-EDA NB.

    Return class estimate 'k', where [0, 1, 2, 3] corresponds to
    [other, square, triangular, sine]"""

    # labels init all with 'other'
    k = np.zeros(full_b.shape[0], dtype=int)

    # square
    k[
        np.all(
            np.abs(full_b[:, 250:500:50] - full_b[:, 200:450:50])
            / np.max(np.abs(full_b), axis=1, keepdims=True)
            < 0.05,
            axis=1,
        )
        & np.all(full_b[:, -200:] < 0, axis=1)
    ] = 1

    # triangular
    k[bool_filter_triangular(full_b, rel_kf=0.01, rel_kc=0.01)] = 2

    # sine
    k[bool_filter_sine(full_b, rel_kf=0.01, rel_kc=0.01)] = 3

    # postprocess "other" signals in frequency-domain, to recover some more squares, triangles, and sines
    n_subsample = 32
    other_b = full_b[k == 0, ::n_subsample]
    other_b /= np.abs(other_b).max(axis=1, keepdims=True)
    other_b_ft = np.abs(np.fft.fft(other_b, axis=1))
    other_b_ft /= other_b_ft.max(axis=1, keepdims=True)
    msk_of_newly_identified_sines = np.all(
        (other_b_ft[:, 3:10] < 0.03) & (other_b_ft[:, [2]] < 0.2), axis=1
    )
    msk_of_newly_identified_triangs = np.all(
        ((other_b_ft[:, 1:8] - other_b_ft[:, 2:9]) > 0), axis=1
    ) | np.all(((other_b_ft[:, 1:8:2] > 1e-2) & (other_b_ft[:, 2:9:2] < 1e-2)), axis=1)
    msk_of_newly_identified_triangs = (
        msk_of_newly_identified_triangs & ~msk_of_newly_identified_sines
    )
    msk_of_newly_identified_squares = np.all(
        (other_b_ft[:, 1:4:2] > 1e-2) & (other_b_ft[:, 2:5:2] < 1e-3), axis=1
    )
    msk_of_newly_identified_squares = (
        msk_of_newly_identified_squares
        & ~msk_of_newly_identified_sines
        & ~msk_of_newly_identified_triangs
    )
    idx_sines = np.arange(k.size)[k == 0][msk_of_newly_identified_sines]
    idx_triangs = np.arange(k.size)[k == 0][msk_of_newly_identified_triangs]
    idx_squares = np.arange(k.size)[k == 0][msk_of_newly_identified_squares]
    k[idx_squares] = 1
    k[idx_triangs] = 2
    k[idx_sines] = 3
    return k


def engineer_features(b_seq, freq, temp, material):
    """Add engineered features to data set"""

    match b_seq:
        case str():
            raise NotImplementedError("b_seq must be an array-like yet")
        case np.ndarray():
            # check b_seq shapes
            match b_seq.ndim:
                case 1:
                    b_seq = b_seq[np.newaxis, :]
                case 2:
                    pass
                case _:
                    raise ValueError(
                        f"Expected b_seq to have either one or two dimensions, but is has {b_seq.ndim}."
                    )

    # maybe resample b_seq to 1024 samples
    if b_seq.shape[-1] != L:
        actual_len = b_seq.shape[-1]
        query_points = np.arange(L)
        support_points = np.arange(actual_len) * L / actual_len
        b_seq = np.row_stack(
            [
                np.interp(query_points, support_points, b_seq[i])
                for i in range(b_seq.shape[0])
            ]
        )

    waveforms = get_waveform_est(b_seq)
    waveforms_df = pd.DataFrame(
        np.zeros((len(waveforms), 4)),
        columns=["wav_other", "wav_square", "wav_triangular", "wav_sine"],
    )
    # one hot encode
    waveform_dummies = pd.get_dummies(waveforms, prefix="wav", dtype=float).rename(
        columns={
            "wav_0": "wav_other",
            "wav_1": "wav_square",
            "wav_2": "wav_triangular",
            "wav_3": "wav_sine",
        }
    )
    for c in waveform_dummies:
        waveforms_df.loc[:, c] = waveform_dummies.loc[:, c]
    ds = pd.DataFrame(b_seq, columns=ALL_B_COLS).assign(
        freq=freq,
        temp=temp,
        material=material,
        **{c: waveforms_df.loc[:, c] for c in waveforms_df},
    )

    dbdt = b_seq[:, 1:] - b_seq[:, :-1]
    b_peak2peak = b_seq.max(axis=1) - b_seq.min(axis=1)

    ds = ds.assign(
        b_peak2peak=b_peak2peak,
        log_peak2peak=np.log(b_peak2peak),
        mean_abs_dbdt=np.mean(np.abs(dbdt), axis=1),
        log_mean_abs_dbdt=np.log(np.mean(np.abs(dbdt), axis=1)),
        sample_time=1 / freq,
    )

    return ds


def construct_tensor_seq2seq(
    df,
    x_cols,
    b_limit,
    h_limit,
    b_limit_pp=None,
    ln_ploss_mean=0,
    ln_ploss_std=1,
    training_data=True,
):
    """Generate tensors with following shapes:
    For time series tensors (#time steps, #profiles/periods, #features),
    for scalar tensors (#profiles, #features)"""
    full_b = df.loc[:, ALL_B_COLS].to_numpy()
    if training_data:
        full_h = df.loc[:, ALL_H_COLS].to_numpy()
    mat = df.iloc[0, :].loc['material']
    df = df.drop(columns=[c for c in df if c.startswith(("H_t_", "B_t_", "material"))])
    assert len(df) > 0, "empty dataframe error"
    # put freq on first place since Architecture expects it there
    x_cols.insert(0, x_cols.pop(x_cols.index("freq")))
    X = df.loc[:, x_cols].astype(np.float32)

    # normalization
    full_b /= b_limit
    if training_data:
        full_h /= h_limit
    orig_freq = X.loc[:, ["freq"]].copy().to_numpy()
    X.loc[:, ["temp", "freq"]] /= np.array([75.0, FREQ_SCALE], dtype=np.float32)
    X.loc[:, "freq"] = np.log(X.freq)
    other_cols = [
        c for c in x_cols if c not in ["temp", "freq"] and not c.startswith("wav_")
    ]
    for other_col in other_cols:
        X.loc[:, other_col] /= NORM_DENOM[mat][other_col]
    
    if training_data:
        # add p loss as target (only used as target when predicting p loss directly), must be last column
        X = X.assign(ln_ploss=(np.log(df.ploss) - ln_ploss_mean) / ln_ploss_std)
    # tensor list
    tens_l = []
    if b_limit_pp is not None:
        # add another B curve with different normalization
        per_profile_scaled_b = full_b * b_limit / b_limit_pp
        # add timeseries derivatives
        b_deriv = np.empty((full_b.shape[0], full_b.shape[1] + 2))
        b_deriv[:, 1:-1] = per_profile_scaled_b
        b_deriv[:, 0] = per_profile_scaled_b[:, -1]
        b_deriv[:, -1] = per_profile_scaled_b[:, 0]
        b_deriv = np.gradient(b_deriv, axis=1) * orig_freq
        b_deriv_sq = np.gradient(b_deriv, axis=1) * orig_freq
        b_deriv = b_deriv[:, 1:-1]
        b_deriv_sq = b_deriv_sq[:, 1:-1]
        tantan_b = -np.tan(0.9 * np.tan(per_profile_scaled_b)) / 6  # tan-tan feature
        tens_l += [
            torch.tensor(per_profile_scaled_b.T[..., np.newaxis], dtype=torch.float32),
            torch.tensor(
                b_deriv.T[..., np.newaxis] / np.abs(b_deriv).max(), dtype=torch.float32
            ),
            torch.tensor(
                b_deriv_sq.T[..., np.newaxis] / np.abs(b_deriv_sq).max(),
                dtype=torch.float32,
            ),
            torch.tensor(tantan_b.T[..., np.newaxis], dtype=torch.float32),
        ]
    tens_l += [
        torch.tensor(full_b.T[..., np.newaxis], dtype=torch.float32)
    ]  # b field is penultimate column
    if training_data:
        tens_l += [
            torch.tensor(
                full_h.T[..., np.newaxis], dtype=torch.float32
            ),  # target is last column
        ]

    # return ts tensor with shape: (#time steps, #profiles, #features), and scalar tensor with (#profiles, #features)
    return torch.dstack(tens_l), torch.tensor(X.to_numpy(), dtype=torch.float32)


class PaderbornModel:
    """The Paderborn model.

    HARDCORE: H-field and power loss estimation for arbitrary waveforms with residual,
     dilated convolutional neural networks in ferrite cores
    N Förster, W Kirchgässner, T Piepenbrock, O Schweins, O Wallscheid
    arXiv preprint arXiv:2401.11488

    """

    def __init__(self, model_path, material):
        self.model_path = model_path
        self.material = material
        self.mdl = torch.jit.load(model_path)
        self.mdl.eval()
        assert (
            material in MAT_CONST_H_MAX and material in MAT_CONST_B_MAX
        ), f"Requested material '{material}' is not supported"
        self.b_limit = MAT_CONST_B_MAX[material]
        self.h_limit = MAT_CONST_H_MAX[material]
        self.predicts_p_directly = True

    def __call__(self, b_seq, frequency, temperature):
        """Evaluate trajectory and estimate power loss.

        Args
        ----
        b_seq: (B, T) array_like
            The magnetic flux density array(s). First dimension describes the batch, the second
             the time length (will always be interpolated to 1024 samples)
        frequency: scalar or 1D array-like
            The frequency operation point(s)
        temperature: scalar or 1D array-like
            The temperature operation point(s)
        """
        ds = engineer_features(b_seq, frequency, temperature, self.material)
        # construct tensors
        x_cols = [
            c
            for c in ds
            if c not in ["ploss", "kfold", "material"]
            and not c.startswith(("B_t_", "H_t_"))
        ]
        b_limit_per_profile = (
            np.abs(ds.loc[:, ALL_B_COLS].to_numpy()).max(axis=1).reshape(-1, 1)
        )
        h_limit = self.h_limit * b_limit_per_profile / self.b_limit
        b_limit_test_fold = self.b_limit
        b_limit_test_fold_pp = b_limit_per_profile
        h_limit_test_fold = h_limit
        with torch.inference_mode():
            val_tensor_ts, val_tensor_scalar = construct_tensor_seq2seq(
                ds,
                x_cols,
                b_limit_test_fold,
                h_limit_test_fold,
                b_limit_pp=b_limit_test_fold_pp,
                training_data=False,
            )

            if self.predicts_p_directly:
                # prepare torch tensors for normalization scales
                b_limit_test_fold_torch = torch.as_tensor(
                    b_limit_test_fold, dtype=torch.float32
                )
                h_limit_test_fold_torch = torch.as_tensor(
                    h_limit_test_fold, dtype=torch.float32
                )
                freq_scale_torch = torch.as_tensor(FREQ_SCALE, dtype=torch.float32)

                val_pred_p, val_pred_h = self.mdl(
                    val_tensor_ts.permute(1, 2, 0),
                    val_tensor_scalar,
                    b_limit_test_fold_torch,
                    h_limit_test_fold_torch,
                    freq_scale_torch,
                )
            else:
                val_pred_h = self.mdl(
                    val_tensor_ts.permute(1, 2, 0),
                    val_tensor_scalar,
                ).permute(2, 0, 1)
                val_pred_p = None
            h_pred = val_pred_h.squeeze().cpu().numpy().T * h_limit_test_fold
            if val_pred_p is None:
                p_pred = frequency * np.trapz(h_pred, b_seq, axis=1)
            else:
                p_pred = np.exp(val_pred_p.squeeze().cpu().numpy())
        return p_pred.astype(np.float32), h_pred.astype(np.float32)
