"""Code is adapted from https://github.com/upb-lea/RHINO-MAG."""

from typing import Callable
from functools import partial
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import equinox as eqx


def filter_spec(f, leaf, f64_enabled):
    """Helper function for proper model loading under varying array precision.

    When a model is saved with float32 arrays and one attempts to load it in a float64
    context (and vice-versa), the loading crashes. This helper transfers the arrays to
    the proper dtypes.

    - float64 is enabled -> convert float32 arrays to float64
    - float64 is disabled -> convert float64 arrays to float32
    """
    problematic_dtype = jnp.float32 if f64_enabled else jnp.float64
    target_dtype = jnp.float64 if f64_enabled else jnp.float32

    if isinstance(leaf, jax.Array):
        loaded_leaf = jnp.load(f)
        if loaded_leaf.dtype == problematic_dtype:
            return loaded_leaf.astype(target_dtype)
        else:
            return loaded_leaf
    else:
        return eqx.default_deserialise_filter_spec(f, leaf)


class Normalizer(eqx.Module):
    """Class used to ease normalization of material data.

    Args:
        B_max (float): Absolute maximum of the B material data used to normalize
        H_max (float): Absolute maximum of the H material data used to normalize
        T_max (float): Absolute maximum of the T material data used to normalize
        norm_fe_max (list[float]): Absolute maximum of the features used to normalize each feature independently
        H_transform (Callable): Optional transform of H
        H_inverse_transform (Callable): Inverse transfrom of the `H_transform`
    """

    B_max: float
    H_max: float
    T_max: float
    norm_fe_max: list[float] = eqx.field(static=True)
    H_transform: callable = eqx.field(static=True)
    H_inverse_transform: callable = eqx.field(static=True)

    def normalize(self, B: jax.Array, H: jax.Array, T: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return (B / self.B_max, self.H_transform(H / self.H_max), T / self.T_max)

    def normalize_H(self, H: jax.Array) -> jax.Array:
        return self.H_transform(H / self.H_max)

    def denormalize(self, B: jax.Array, H: jax.Array, T: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        H = self.H_inverse_transform(H)
        return B * self.B_max, H * self.H_max, T * self.T_max

    def denormalize_H(self, H: jax.Array) -> jax.Array:
        H = self.H_inverse_transform(H)
        return H * self.H_max

    def normalize_fe(self, features: jax.Array) -> jax.Array:
        fe_norm = features / jnp.array(self.norm_fe_max)
        return fe_norm

    def denormalize_fe(self, features: jax.Array) -> jax.Array:
        return features * jnp.array(self.norm_fe_max)

    @classmethod
    def from_dict(cls, params: dict):
        """Creates a normalizer from a dict for easy loading from disk.

        NOTE: Assumes a tanh transformation as the only possible transformation.
        """

        if params["transform_H"] == False:
            params["H_transform"] = lambda x: x
            params["H_inverse_transform"] = lambda x: x
        elif params["transform_H"] == True:
            params["H_transform"] = lambda h: jnp.tanh(h * 1.2)
            params["H_inverse_transform"] = lambda h: jnp.atanh(h) / 1.2
        del params["transform_H"]
        return cls(**params)

    def to_dict(self, transform_H: bool = False):
        """Creates a dict from the normalizer for easy storing on disk."""
        params = {}
        params["transform_H"] = transform_H
        params["B_max"] = self.B_max
        params["H_max"] = self.H_max
        params["T_max"] = self.T_max
        params["norm_fe_max"] = self.norm_fe_max
        return params


def setup_featurize(
    disable_features: bool,
    dyn_avg_kernel_size: int,
    time_shift: int,
) -> Callable:
    """Setup the `featurize` function.

    Args:
        disable_features (bool | str): One of (True, False, "reduce"), True uses no features, False uses all default features,
            "reduce" uses the dB/dt and d^2 B / dt^2 as features.
        dyn_avg_kernel_size (int): The kernel size of the dynamic average feature.
        time_shift (int): When specifying a value `!=0`, a feature is added where the `B` trajectory is shifted by that
            number of time steps

    Returns:
        Function that adds features to the input material data

    """

    def db_dt(b: jax.Array) -> jax.Array:
        """Calculate the first derivative of b."""
        return jnp.gradient(b)

    def d2b_dt2(b: jax.Array) -> jax.Array:
        """Calculate the first derivative of b."""
        return jnp.gradient(jnp.gradient(b))

    if disable_features == "reduce":

        def featurize(norm_B_past, norm_H_past, norm_B_future, temperature, time_shift):
            past_length = norm_B_past.shape[0]
            B_all = jnp.hstack([norm_B_past, norm_B_future])
            db = db_dt(B_all)
            d2b = d2b_dt2(B_all)
            featurized_B = jnp.stack((db, d2b), axis=-1)
            return featurized_B[past_length:]

    else:
        raise ValueError("Option 'disable_features' with value '{disable_features}' cannot be processed.")
    featurize = partial(featurize, time_shift=time_shift)
    return featurize


#### Model


class GRU(eqx.Module):
    """Basic gated recurrent unit (GRU) model."""

    hidden_size: int = eqx.field(static=True)
    cell: eqx.Module

    def __init__(self, in_size: int, hidden_size: int, *, key):
        """Construct a basic Gated Recurrent Unit (GRU) based on the `equinox.nn.GRUCell`.

        Args:
            in_size (int): Number of input elements
            hidden_size (int): Number of hidden state elements
            key (jax.random.PRNGkey): Pseudo random number generation key for initialization of the
                model parameters
        """
        self.hidden_size = hidden_size
        self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=key)

    def __call__(self, input: jax.Array, init_hidden: jax.Array) -> jax.Array:
        """Use the GRU to roll over an input sequence.

        The first element of the hidden state is interpreted as the output of the GRU in each.

        NOTE: This function is not expecting a batch dimension. It is intended to vmapped over for
            batch-wise predictions!

        Args:
            input (jax.Array): Input sequence with shape (sequence_length, in_size)
            init_hidden (jax.Array): Initial vector for the hidden state with shape (hidden_size,)

        Returns:
            The output sequence as a jax.Array with shape (sequence_length, 1)
        """

        hidden = init_hidden

        def f(carry, inp):
            rnn_out = self.cell(inp, carry)
            rnn_out_o = jnp.atleast_2d(rnn_out)
            out = rnn_out_o[..., 0]
            return rnn_out, out

        _, out = jax.lax.scan(f, hidden, input)
        return out

    def warmup_call(self, input: jax.Array, init_hidden: jax.Array, out_true: jax.Array) -> jax.Array:
        """Warm up the hidden state of the GRU with an input sequence where the true outputs are known.

        The basic idea is to feed the true value into the first element of the hidden state in each step
        so that the output prediction does not diverge while the other elements of the GRU may change to
        get into shape to best predict the following sequence without known true values.

        Args:
            input (jax.Array): Input sequence with shape (sequence_length, in_size)
            init_hidden (jax.Array): Initial vector for the hidden state with shape (hidden_size,)
            out_true (jax.Array): The true output values with shape (sequence_length,)

        Returns:
            The outputs as jax.Array with shape (sequence_length,) (these will be the same as out_true) and
                the final warmed up hidden_state
        """
        hidden = init_hidden

        def f(carry, inp):
            inp_t, out_true_t = inp
            rnn_out = self.cell(inp_t, carry)
            rnn_out = rnn_out.at[0].set(out_true_t)
            rnn_out_o = jnp.atleast_2d(rnn_out)
            out = rnn_out_o[..., 0]
            return rnn_out, out

        final_hidden, out = jax.lax.scan(f, hidden, (input, out_true))
        return out, final_hidden

    def construct_init_hidden(self, out_true: jax.Array, batch_size: int) -> jax.Array:
        """Put together the very first initial state. Concatenates the given true value with zeros."""
        return jnp.hstack([out_true, jnp.zeros((batch_size, self.hidden_size - 1))])


class ModelInterface(eqx.Module):

    @abstractmethod
    def __call__(
        self,
        B_past: jax.Array,
        H_past: jax.Array,
        B_future: jax.Array,
        T: jax.Array,
    ) -> jax.Array:
        """Model prediction interface for batched inputs, i.e. for inputs with an extra
        leading dimension.

        Args:
            B_past (jax.Array): The physical (non-normalized) flux density values from time
                step k0 to k1 with shape (n_batches, past_sequence_length)
            H_past (jax.Array): The physical (non-normalized) field values from time step
                k0 to k1 with shape (n_batches, past_sequence_length)
            B_future (jax.Array): The physical (non-normalized) flux density values from
                time step k1 to k2 with shape (n_batches, future_sequence_length)
            T (float): The temperature of the material with shape (n_batches,)

        Returns:
            H_pred (jax.Array): The physical (non-normalized) field values from time
                step k1 to k2 with shape (n_batches, future_sequence_length)
        """
        pass

    @abstractmethod
    def normalized_call(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        """
        Model prediction interface for normalized, batched inputs, i.e. for inputs with
        an extra leading dimension.

        Args:
            B_past_norm (jax.Array): The normalized flux density values from time step k0
                to k1 with shape (n_batches, past_sequence_length)
            H_past_norm (jax.Array): The normalized field values from time step k0 to k1
                with shape (n_batches, past_sequence_length)
            B_future_norm (jax.Array): The physical normalized flux density values from
                time step k1 to k2 with shape (n_batches, future_sequence_length)
            T_norm (float): The normalized temperature of the material with shape (n_batches,)

        Returns:
            H_pred_norm (jax.Array): The normalized field values from time step k1 to k2
                with shape (n_batches, future_sequence_length)
        """
        pass


class RNNwInterface(ModelInterface):
    """Model interface for basic recurrent neural network (RNN) models.

    Manages the interaction between generic data-driven model and input material data.

    Args:
        model (GRU): The generic RNN model
        normalizer (Normalizer): The normalization object to normalize the raw material data for the RNN
        featurize (Callable): The featurization function to add further features to the material data.
    """

    model: GRU
    normalizer: Normalizer
    featurize: Callable = eqx.field(static=True)

    def __call__(
        self,
        B_past: jax.Array,
        H_past: jax.Array,
        B_future: jax.Array,
        T: jax.Array,
        warmup: bool = True,
    ) -> jax.Array:
        """Main function to be called for this model object.

        Takes the material data and produces the model prediction. In between, the data is
        normalized, featurized, a warmup of the hidden state of the RNN is performed, the
        H prediction is performed and the output is denormalized back to a physical value,
        i.e., from a numerical value back to its interpretation as `Ampere/m`.

        Args:
            B_past (jax.Array): The physical (non-normalized) flux density values from time
                step k0 to k1 with shape (n_batches, past_sequence_length)
            H_past (jax.Array): The physical (non-normalized) field values from time step
                k0 to k1 with shape (n_batches, past_sequence_length)
            B_future (jax.Array): The physical (non-normalized) flux density values from
                time step k1 to k2 with shape (n_batches, future_sequence_length)
            T (float): The temperature of the material with shape (n_batches,)

        Returns:
            H_pred (jax.Array): The physical (non-normalized) field values from time
                step k1 to k2 with shape (n_batches, future_sequence_length)
        """

        B_all = jnp.concatenate([B_past, B_future], axis=1)
        B_all_norm, H_past_norm, T_norm = self.normalizer.normalize(B_all, H_past, T)

        B_past_norm = B_all_norm[:, : B_past.shape[1]]
        B_future_norm = B_all_norm[:, B_past.shape[1] :]

        batch_H_pred = self.normalized_call(B_past_norm, H_past_norm, B_future_norm, T_norm, warmup)
        batch_H_pred_denorm = jax.vmap(jax.vmap(self.normalizer.denormalize_H))(batch_H_pred)

        return batch_H_pred_denorm

    def _prepare_model_input(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        """Prepare the input vector for the model based on the provided material data.

        Args:
            B_past_norm (jax.Array): The normalized flux density values from time step k0
                to k1 with shape (n_batches, past_sequence_length)
            H_past_norm (jax.Array): The normalized field values from time step k0 to k1
                with shape (n_batches, past_sequence_length)
            B_future_norm (jax.Array): The physical normalized flux density values from
                time step k1 to k2 with shape (n_batches, future_sequence_length)
            T_norm (float): The normalized temperature of the material with shape (n_batches,)

        Returns:
            batch_x (jax.Array): The input to the RNN with shape (n_batches, future_sequence_length, gru_in_size)

        """
        features = jax.vmap(self.featurize, in_axes=(0, 0, 0, 0))(B_past_norm, H_past_norm, B_future_norm, T_norm)
        features_norm = jax.vmap(jax.vmap(self.normalizer.normalize_fe))(features)

        T_norm_broad = jnp.broadcast_to(T_norm[:, None], B_future_norm.shape)

        batch_x = jnp.concatenate([B_future_norm[..., None], T_norm_broad[..., None], features_norm], axis=-1)
        return batch_x

    def _warmup(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
    ) -> jax.Array:
        """Warm-up the hidden state of the RNN based on the previous trajectory data.

        The warmup process is essentially a prediction process where the first element of H_past
        is used to initialize the first hidden state and where the first element of the hidden state
        is corrected with the other true H_past value after each step.

        NOTE: The future values of B are actually not used here but only passed for alignment with the
        other interfaces.

        Args:
            B_past_norm (jax.Array): The normalized flux density values from time step k0
                to k1 with shape (n_batches, past_sequence_length)
            H_past_norm (jax.Array): The normalized field values from time step k0 to k1
                with shape (n_batches, past_sequence_length)
            B_future_norm (jax.Array): The physical normalized flux density values from
                time step k1 to k2 with shape (n_batches, future_sequence_length)
            T_norm (float): The normalized temperature of the material with shape (n_batches,)

        Returns:
            final_hidden_warmup (jax.Array): The warmed up hidden state.
        """

        batch_x = self._prepare_model_input(B_past_norm, H_past_norm, B_past_norm, T_norm)
        batch_x = batch_x[:, 1:]

        init_hidden = self.model.construct_init_hidden(
            out_true=H_past_norm[:, 0, None],
            batch_size=H_past_norm.shape[0],
        )
        _, final_hidden_warmup = jax.vmap(self.model.warmup_call)(batch_x, init_hidden, H_past_norm[:, 1:])
        return final_hidden_warmup

    def normalized_call(
        self,
        B_past_norm: jax.Array,
        H_past_norm: jax.Array,
        B_future_norm: jax.Array,
        T_norm: jax.Array,
        warmup: bool = True,
    ) -> jax.Array:
        """Performs the warmup and the prediction on the normalized data.

        Args:
            B_past_norm (jax.Array): The normalized flux density values from time step k0
                to k1 with shape (n_batches, past_sequence_length)
            H_past_norm (jax.Array): The normalized field values from time step k0 to k1
                with shape (n_batches, past_sequence_length)
            B_future_norm (jax.Array): The physical normalized flux density values from
                time step k1 to k2 with shape (n_batches, future_sequence_length)
            T_norm (float): The normalized temperature of the material with shape (n_batches,)
            warmup (bool): Whether warmup should be performed or only the initial state should
                be constructed filled with the first true value and zeros.

        Returns:
            The normalized field prediction as a jax.Array with shape (n_batches, future_sequence_length)
        """
        if warmup and H_past_norm.shape[1] > 1:
            init_hidden = self._warmup(B_past_norm, H_past_norm, B_future_norm, T_norm)
        else:
            init_hidden = self.model.construct_init_hidden(
                out_true=H_past_norm[:, -1, None],
                batch_size=H_past_norm.shape[0],
            )

        batch_x = self._prepare_model_input(B_past_norm, H_past_norm, B_future_norm, T_norm)
        batch_H_pred = jax.vmap(self.model)(batch_x, init_hidden)
        return batch_H_pred[:, :, 0]


def setup_model(
    model_label: str,
    model_key: jax.random.PRNGKey,
    normalizer: Normalizer,
    featurize: Callable,
    time_shift: int = 0.0,
) -> tuple[ModelInterface, dict]:
    """
    Creates the model wrapped into its model interface from the provided parameterization.

    Args:
        model_label (str): Identifier of the model types to be created.
        model_key (jax.random.PRNGKey): Pseudo random number generation key for the creation of the model.
            This key is derived from the key initially given into the algorithm, if `setup_model` is used
            from within the training script.
        normalizer (Normalizer): Normalizer object used to normalize the material data.
        featurize (Callable): Featurize function used to add features to the input of the model.
        time_shift (int): When specifying a value `!=0`, a feature is added where the `B` trajectory is shifted by that
            number of time steps

    Returns:
        The ModelInterface (i.e. the wrapped model) and the model parameterization as a dict
    """

    # dynamically choose model input size:
    test_seq_length = 100
    test_out = featurize(
        norm_B_past=jnp.ones(test_seq_length),
        norm_H_past=jnp.ones(test_seq_length),
        norm_B_future=jnp.ones(test_seq_length),
        temperature=jnp.ones(1),
        time_shift=time_shift,
    )
    assert test_out.shape[0] == test_seq_length
    model_in_size = test_out.shape[-1] + 2  # (+2) due to: flux density B and temperature T

    if model_label == "GRU8":
        hidden_size = 8
        model_params_d = dict(hidden_size=hidden_size, in_size=model_in_size, key=model_key)
        model = GRU(**model_params_d)
        mdl_interface_cls = RNNwInterface

    wrapped_model = mdl_interface_cls(
        model=model,
        normalizer=normalizer,
        featurize=featurize,
    )

    return wrapped_model, model_params_d
