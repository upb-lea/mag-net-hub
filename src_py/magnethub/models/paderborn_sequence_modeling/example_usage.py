"""Code is adapted from https://github.com/upb-lea/RHINO-MAG."""

import pathlib
import json
from functools import partial
from copy import deepcopy

import jax
import jax.numpy as jnp
import equinox as eqx

from helper_functions import filter_spec, Normalizer, setup_featurize, setup_model


def reconstruct_model_from_file(filename: pathlib.Path):
    """Reconstruct a model from its file stored on disk.

    Args:
        filename (pathlib.Path): The path of file to load. If the suffix is missing, `.eqx` is
            added by default. One can  give the full path of the model file or, if the
            file cannot be found at the specified path, the function looks for the model in
            the `mc2.data_management.MODEL_DUMP_ROOT`, which is the default folder models are
            stored in.

            The commands:
            ```
            from mc2.data_management import MODEL_DUMP_ROOT
            reconstruct_model_from_file('A_GRU8_reduced-features-f32_2a1473b6_seed12')
            reconstruct_model_from_file('A_GRU8_reduced-features-f32_2a1473b6_seed12.eqx')
            reconstruct_model_from_file(MODEL_DUMP_ROOT / 'A_GRU8_reduced-features-f32_2a1473b6_seed12.eqx')
            ```
            All load the same model at `data/models` unless a file with the same name is present in the cwd
            in which case the first two calls load that model from the cwd and the last one loads the model
            from `data/models`. Generally, the intended way is to call using the first version.

    Returns:
        The ModelInterface object (i.e., the model wrapped into the corresponing interface)
    """

    filename = pathlib.Path(filename)

    # append the '.eqx' suffix if it is missing
    if filename.suffix == "":
        filename = filename.with_name(f"{filename.name}.eqx")

    with open(filename, "rb") as f:
        params = json.loads(f.readline().decode())

        normalizer = Normalizer.from_dict(params["normalizer_dict"])
        featurize = setup_featurize(
            disable_features=params["training_params"]["disable_features"],
            dyn_avg_kernel_size=params["training_params"]["dyn_avg_kernel_size"],
            time_shift=params["training_params"]["time_shift"],
        )
        fresh_wrapped_model, _ = setup_model(
            model_label=params["model_type"],
            model_key=jax.random.PRNGKey(0),
            normalizer=normalizer,
            featurize=featurize,
            time_shift=params["training_params"]["time_shift"],
        )

        loading_params = deepcopy(params["model_params"])
        loading_params["key"] = jnp.array(loading_params["key"], dtype=jnp.uint32)
        try:
            model = type(fresh_wrapped_model.model)(**loading_params)
        except TypeError:
            model = type(fresh_wrapped_model.model)(normalizer=normalizer, **loading_params)
        model = eqx.tree_deserialise_leaves(f, model, partial(filter_spec, f64_enabled=jax.config.x64_enabled))
        wrapped_model = eqx.tree_at(lambda t: t.model, fresh_wrapped_model, model)

    return wrapped_model


if __name__ == "__main__":

    # load model
    model = reconstruct_model_from_file("3C90_GRU8_MagNetHub-reduced-features-f32_5c3f8051_seed49")

    # generate dummy data
    batch_size = 20
    sequence_length = 1000
    warmup_len = 100

    key = jax.random.PRNGKey(seed=0)
    b_key, h_key, t_key = jax.random.split(key, 3)

    B_sequence = jax.random.normal(b_key, (batch_size, sequence_length))  # (20, 1000)
    H_past = jax.random.normal(h_key, (batch_size, warmup_len))  # (20, 100)
    T = jax.random.randint(t_key, (batch_size,), minval=25, maxval=70)

    # inference
    H_pred = model(
        B_past=B_sequence[:, :warmup_len],  # (20, 100)
        H_past=H_past,  # (20, 100)
        B_future=B_sequence[:, warmup_len:],  # (20, 900)
        T=T,  # (20,)
    )
    print(H_pred.shape)  # (20, 900)
