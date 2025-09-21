import contextlib
import functools
from typing import Callable, List, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from sae_refusal import clear_memory


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs,
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn


def get_direction_ablation_input_pre_hook_rmu(directions: List[Tensor]):
    def hook_fn(module, input):
        nonlocal directions

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        for direction in directions:
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = direction.to(activation)
            activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn


def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def get_direction_ablation_output_hook_rmu(directions: List[Tensor]):
    def hook_fn(module, input, output):
        nonlocal directions

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        for direction in directions:
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = direction.to(activation)
            activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, "d_model"],
):
    fwd_pre_hooks = [
        (
            model_base.model_block_modules[layer],
            get_direction_ablation_input_pre_hook(direction=direction),
        )
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks = [
        (
            model_base.model_attn_modules[layer],
            get_direction_ablation_output_hook(direction=direction),
        )
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks += [
        (
            model_base.model_mlp_modules[layer],
            get_direction_ablation_output_hook(direction=direction),
        )
        for layer in range(model_base.model.config.num_hidden_layers)
    ]

    return fwd_pre_hooks, fwd_hooks


def get_all_direction_ablation_hooks_rmu(
    model_base,
    directions: List[Float[Tensor, "d_model"]],
):
    fwd_pre_hooks = [
        (
            model_base.model_block_modules[layer],
            get_direction_ablation_input_pre_hook_rmu(directions=directions),
        )
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks = [
        (
            model_base.model_attn_modules[layer],
            get_direction_ablation_output_hook_rmu(directions=directions),
        )
        for layer in range(model_base.model.config.num_hidden_layers)
    ]
    fwd_hooks += [
        (
            model_base.model_mlp_modules[layer],
            get_direction_ablation_output_hook_rmu(directions=directions),
        )
        for layer in range(model_base.model.config.num_hidden_layers)
    ]

    return fwd_pre_hooks, fwd_hooks


def get_directional_patching_input_pre_hook(
    direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]
):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn


def get_activation_addition_input_pre_hook(
    vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]
):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn


def get_activation_addition_input_pre_hook_rmu(
    vectors: List[Float[Tensor, "d_model"]], coeffs: List[Float[Tensor, ""]]
):
    def hook_fn(module, input):
        nonlocal vectors

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        for vector, coeff in zip(vectors, coeffs):
            vector = vector.to(activation)
            activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation

    return hook_fn


def get_activations_pre_hook(
    cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]
):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = (
            input[0].clone().to(cache)
        )
        cache[:, :] += activation[:, positions, :]

    return hook_fn


def get_activations_hook(
    cache: Float[Tensor, "pos layer d_model"], positions: List[int]
):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # output: [batch_size, seq_len, d_model]
        cache[:, :] += activation.clone().to(cache)[:, positions, :]

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn
