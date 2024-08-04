"""
File contains the model according to the Sydney University approach for the magnet challenge.

Source: https://github.com/moetomg/magnet-engine
"""

import torch
import numpy as np
from scipy.signal import savgol_filter

MAT2FILENAME = {
    "3C90": "3C90.pt",
    "3C92": "3C92.pt",
    "3C94": "3C94.pt",
    "3C95": "3C95.pt",
    "3E6": "3E6.pt",
    "3F4": "3F4.pt",
    "77": "77.pt",
    "78": "78.pt",
    "79": "79.pt",
    "T37": "T37.pt",
    "N27": "N27.pt",
    "N30": "N30.pt",
    "N49": "N49.pt",
    "N87": "N87.pt",
    "ML95S": "ML95S.pt",
}
# Material normalization data (1.B 2.H 3.F 4.T 5.dB/dt)
normsDict = {
    "77": [
        [-2.63253458e-19, 7.47821754e-02],
        [-7.60950004e-18, 1.10664739e01],
        [5.24678898e00, 2.89351404e-01],
        [5.87473793e01, 2.40667381e01],
        [6.16727829e00, 3.83645439e01],
    ],
    "78": [
        [5.67033925e-19, 7.22424510e-02],
        [-1.54283684e-16, 1.15338828e01],
        [5.23810768e00, 2.89979160e-01],
        [5.87434082e01, 2.40685291e01],
        [6.09561586e00, 3.81356049e01],
    ],
    "79": [
        [1.70344847e-13, 9.41321492e-02],
        [-4.54025068e-02, 3.20463941e01],
        [5.21954346e00, 2.66715437e-01],
        [5.52068787e01, 2.37196522e01],
        [6.77422905e00, 3.90895233e01],
    ],
    "N27": [
        [7.52738469e-19, 7.48951129e-02],
        [-8.97477366e-17, 1.47606605e01],
        [5.24649334e00, 2.89964765e-01],
        [5.87355194e01, 2.40766029e01],
        [6.17841434e00, 3.84738274e01],
    ],
    "N30": [
        [1.43320465e-19, 6.56044649e-02],
        [-1.57874135e-16, 1.09083332e01],
        [5.31786680e00, 2.78960317e-01],
        [5.86466904e01, 2.40616817e01],
        [7.01255989e00, 4.09709969e01],
    ],
    "N49": [
        [-8.99073580e-19, 8.94479227e-02],
        [4.15423721e-16, 3.70622618e01],
        [5.25545311e00, 3.00384015e-01],
        [5.94716339e01, 2.44349327e01],
        [6.75209475e00, 3.91901703e01],
    ],
    "N87": [
        [1.72051200e-13, 6.26231476e-02],
        [4.02299992e-02, 7.61060358e00],
        [5.26309967e00, 2.87137657e-01],
        [5.83059006e01, 2.40639057e01],
        [6.53078842e00, 3.93127785e01],
    ],
    "3E6": [
        [1.01579639e-18, 7.04261607e-02],
        [2.34374135e-16, 7.21573964e00],
        [5.34307003e00, 2.66708523e-01],
        [5.86578026e01, 2.40552864e01],
        [7.23155785e00, 4.15975838e01],
    ],
    "3F4": [
        [-1.75200068e-19, 5.98892952e-02],
        [-9.48865199e-18, 4.74414811e01],
        [5.14398336e00, 3.04210454e-01],
        [5.76523476e01, 2.43824081e01],
        [6.23030663e00, 3.64991379e01],
    ],
    "T37": [
        [1.72051200e-13, 6.26231476e-02],
        [4.02299992e-02, 7.61060358e00],
        [5.26309967e00, 2.87137657e-01],
        [5.83059006e01, 2.40639057e01],
        [6.53078842e00, 3.93127785e01],
    ],
    "3C90": [
        [-3.27923689e-19, 6.56109348e-02],
        [6.99196716e-17, 1.26583787e01],
        [5.19875193e00, 2.68499136e-01],
        [5.86049919e01, 2.40574703e01],
        [6.29652929e00, 3.84585190e01],
    ],
    "3C92": [
        [-2.35520104e-13, 6.53518693e-02],
        [1.18689366e-01, 1.23585692e01],
        [5.16579533e00, 2.73998171e-01],
        [5.84305267e01, 2.40970516e01],
        [5.88209248e00, 3.69935722e01],
    ],
    "3C94": [
        [1.21232679e-19, 7.44383659e-02],
        [-2.19613879e-17, 1.18042579e01],
        [5.22766781e00, 2.68348873e-01],
        [5.87128143e01, 2.40769634e01],
        [6.53718996e00, 3.91955910e01],
    ],
    "3C95": [
        [5.64116728e-14, 7.90115297e-02],
        [1.11898437e-01, 1.29696641e01],
        [5.18842697e00, 2.69014776e-01],
        [5.86223640e01, 2.40957470e01],
        [6.25767517e00, 3.84026108e01],
    ],
    "ML95S": [
        [-1.53185180e-13, 1.15827541e-01],
        [3.84426934e-01, 4.45061606e01],
        [5.21606445e00, 2.65364528e-01],
        [5.70770302e01, 2.44398289e01],
        [7.30377579e00, 4.04136391e01],
    ],
}


# %% Initialize model
class SydneyModel:
    """The Sydney model."""

    expected_seq_len = 128  # the expected sequence length

    def __init__(self, mdl_path, material):
        # Select GPU as default device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1.Create model isntances
        self.mdl = MMINet(material).to(self.device)

        # 2.Load specific model
        state_dict = torch.load(mdl_path, map_location=self.device)
        self.mdl.load_state_dict(state_dict, strict=True)

    def __call__(self, data_B, data_F, data_T):
        """Call method."""
        # ----------------------------------------------------------- batch execution
        # 1.Get dataloader
        if data_B.ndim == 1:
            data_B = np.array(data_B).reshape(1, -1)

        _, ts_feats, scalar_feats = get_dataloader(data_B, data_F, data_T, self.mdl.norm)

        # 2.Validate the models
        self.mdl.eval()
        with torch.inference_mode():
            # Start model evaluation explicitly
            data_P, h_series = self.mdl(ts_feats.to(self.device), scalar_feats.to(self.device))

        data_P, h_series = data_P.cpu().numpy(), h_series.cpu().numpy()

        # 3.Return results
        if data_P.size == 1:
            data_P = data_P.item()
        if h_series.ndim == 1:
            h_series = h_series.reshape(1, -1)

        return data_P, h_series


class MMINet(torch.nn.Module):
    """
    Magnetization mechanism-determined neural network.

    Parameters:
    - hidden_size: number of eddy current slices (RNN neuron)
    - operator_size: number of operators
    - input_size: number of inputs (1.B 2.dB 3.dB/dt)
    - var_size: number of supplenmentary variables (1.F 2.T)
    - output_size: number of outputs (1.H)
    """

    def __init__(self, Material, hidden_size=30, operator_size=30, input_size=3, var_size=2, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.var_size = var_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.operator_size = operator_size
        self.norm = normsDict[Material]  # normalization data
        self.n_init = 32

        # Consturct the network
        self.rnn1 = StopOperatorCell(self.operator_size)
        self.dnn1 = torch.nn.Linear(self.operator_size + 2, 1)
        self.rnn2 = EddyCell(4, self.hidden_size, output_size)
        self.dnn2 = torch.nn.Linear(self.hidden_size, 1)

        self.rnn2_hx = None

    def forward(self, x, var):
        """
        Forward function.

        Parameters:
        x: batch,seq,input_size
            Input features (1.B, 2.dB, 3.dB/dt)
        var: batch,var_size
            Supplementary inputs (1.F 2.T)
        """
        batch_size = x.size(0)  # Batch size
        seq_size = x.size(1)  # Ser

        # Initialize operator state
        self.rnn1_hx = var[:, 2:]

        # Initialize DNN2 input (1.B 2.dB/dt)
        x2 = torch.cat((x[:, :, 0:1], x[:, :, 2:3]), dim=2)
        for t in range(seq_size):
            # RNN1 input (dB,state)
            self.rnn1_hx = self.rnn1(x[:, t, 1:2], self.rnn1_hx)

            # DNN1 input (rnn1_hx,F,T)
            dnn1_in = torch.cat((self.rnn1_hx, var[:, 0:2]), dim=1)

            # H hysteresis prediction
            H_hyst_pred = self.dnn1(dnn1_in)

            # DNN2 input (B,dB/dt,T,F)
            rnn2_in = torch.cat((x2[:, t, :], var[:, 0:2]), dim=1)

            # Initialize second rnn state
            if t == 0:
                H_eddy_init = x[:, t, 0:1] - H_hyst_pred
                buffer = x.new_ones(x.size(0), self.hidden_size)
                self.rnn2_hx = torch.autograd.Variable((buffer / torch.sum(self.dnn2.weight, dim=1)) * H_eddy_init)

            self.rnn2_hx = self.rnn2(rnn2_in, self.rnn2_hx)

            # H eddy prediction
            H_eddy = self.dnn2(self.rnn2_hx)

            # H total
            H_total = (H_hyst_pred + H_eddy).view(batch_size, 1, self.output_size)

            if t == 0:
                output = H_total
            else:
                output = torch.cat((output, H_total), dim=1)

        # Compute the power loss density
        B = x[:, self.n_init :, 0:1] * self.norm[0][1] + self.norm[0][0]
        H = output[:, self.n_init :, :] * self.norm[1][1] + self.norm[1][0]
        Pv = torch.trapz(H, B, axis=1) * (10 ** (var[:, 0:1] * self.norm[2][1] + self.norm[2][0]))

        # Return results
        H = savgol_filter(H.detach().to("cpu").numpy(), window_length=7, polyorder=2, axis=1)
        H = torch.from_numpy(H).view(batch_size, -1, 1)
        real_H = torch.cat((H[:, -self.n_init :, :], H[:, : -self.n_init, :]), dim=1)
        return torch.flatten(Pv).cpu(), real_H[:, :, 0].cpu()


class StopOperatorCell:
    """
    MMINN Sub-layer: Static hysteresis prediction using stop operators.

    Parameters:
    - operator_size: number of operator
    """

    def __init__(self, operator_size):
        self.operator_thre = (
            torch.pow(
                torch.arange(1, operator_size + 1, dtype=torch.float) / (operator_size + 1), torch.tensor(3.0)
            ).view(1, -1)
            * 1
        )

    def sslu(self, X):
        """Hardsigmoid-like or symmetric saturated linear unit definition."""
        a = torch.ones_like(X)
        return torch.max(-a, torch.min(a, X))

    def __call__(self, dB, state):
        """Update operator of each time step."""
        r = self.operator_thre.to(dB.device)
        output = self.sslu((dB + state) / r) * r
        return output.float()


class EddyCell(torch.nn.Module):
    """
    MMINN subsubnetwork: Dynamic hysteresis prediction.

    Parameters:
    - input_size: feature size
    - hidden_size: number of hidden units (eddy current layers)
    - output_size: number of the output
    """

    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.x2h = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, hidden=None):
        """
        Forward function.

        Parameters:
        x: batch,input_size
            features (1.B 2.dB/dt 3.F 4.T)
        hidden: batch,hidden_size
            dynamic hysteresis effects at each eddy current layer
        """
        hidden = self.x2h(x) + self.h2h(hidden)
        hidden = torch.sigmoid(hidden)
        return hidden


def get_dataloader(data_B, data_F, data_T, norm, n_init=32):
    """Preprocess data into a data loader.
    Get a test dataloader.

    Parameters
    ---------
    data_B: array
         B data
    data_F
         F data
    data_T
         T data
    norm : list
         B/F/T normalization data
    n_init : int
         Additional points for computing the history magnetization
    """

    # Data pre-process
    # 1. Down-sample to 128 points
    seq_length = 128

    if data_B.shape[-1] != seq_length:
        cols = np.array(range(0, data_B.shape[1], round(data_B.shape[1] / seq_length)))
        data_B = data_B[:, cols]

    # 2. Add extra points for initial magnetization calculation
    data_length = seq_length + n_init
    data_B = np.hstack((data_B, data_B[:, 1 : 1 + n_init]))

    # 3. Format data into tensors
    B = torch.from_numpy(data_B).view(-1, data_length, 1).float()
    if np.isscalar(data_F):
        data_F = np.array([data_F])
    if np.isscalar(data_T):
        data_T = np.array([data_T])
    T = torch.from_numpy(data_T).view(-1, 1).float()
    F = torch.from_numpy(np.log10(data_F)).view(-1, 1).float()

    # 4. Data Normalization
    in_B = (B - norm[0][0]) / norm[0][1]
    in_F = (F - norm[2][0]) / norm[2][1]
    in_T = (T - norm[3][0]) / norm[3][1]

    # 5. Extra features
    in_dB = torch.diff(in_B, dim=1)  # Flux density change
    in_dB = torch.cat((in_dB[:, 0:1, :], in_dB), dim=1)

    dB_dt = in_dB * (seq_length * F.reshape(-1, 1, 1))
    in_dB_dt = (dB_dt - norm[4][0]) / norm[4][1]  # Flux density change rate

    max_B, _ = torch.max(in_B, dim=1)
    min_B, _ = torch.min(in_B, dim=1)

    s0 = get_operator_init(in_B[:, 0, 0] - in_dB[:, 0, 0], in_dB, max_B, min_B)  # Operator inital state

    ts_feats = torch.cat((in_B, in_dB, in_dB_dt), dim=2)
    scalar_feats = torch.cat((in_F, in_T, s0), dim=1)
    # 6. Create dataloader to speed up data processing
    test_dataset = torch.utils.data.TensorDataset(ts_feats, scalar_feats)
    kwargs = {"num_workers": 0, "batch_size": 128, "drop_last": False}
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    return test_loader, ts_feats, scalar_feats


# %% Predict the operator state at t0
def get_operator_init(B0, dB, Bmax, Bmin, operator_size=30, max_out_H=1):
    """Compute the initial state of hysteresis operators.

    Parameters
    ---------
    B0 : torch_like (batch)
         Stop operator excitation at t1
    dB : torch_like (batch, data_length)
         Flux density changes at each t
    Bmax: torch_like (batch)
         Max flux density of each cycle
    Bmin: torch_like (batch)
         Min flux density of each cycle
    operator_size: int
         The number of operators
    max_out_H:
         The maximum output of field strength
    """
    # 1. Parameter setting
    batch = dB.shape[0]
    state = torch.zeros((batch, operator_size))
    operator_thre = (
        torch.pow(torch.arange(1, operator_size + 1, dtype=torch.float) / operator_size + 1, torch.tensor(3.0)).view(
            1, -1
        )
        * max_out_H
    )

    # 2. Iterate each excitation for the operator inital state computation
    for i in range(B0.__len__()):
        for j in range(operator_size):
            r = operator_thre[0, j]
            if (Bmax[i] >= r) or (Bmin[i] <= -r):
                if dB[i, 0] >= 0:
                    if B0[i] > Bmin[i] + 2 * r:
                        state[i, j] = r
                    else:
                        state[i, j] = B0[i] - (r + Bmin[i])
                else:
                    if B0[i] < Bmax[i] - 2 * r:
                        state[i, j] = -r
                    else:
                        state[i, j] = B0[i] + (r - Bmax[i])

    return state
