import contextlib
import functools
import random
from typing import Callable, Dict, List, Optional, Tuple

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm


def get_sae_fwd_pre_hook(sae, reconstruct_bos_token: bool = False):
    def hook_fn(
        module,
        input,
        input_ids: Int[Tensor, "batch_size seq_len"] = None,
        attention_mask: Int[Tensor, "batch_size seq_len"] = None,
    ):
        nonlocal sae, reconstruct_bos_token

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        dtype = activation.dtype
        batch_size, seq_pos, d_model = activation.shape

        reconstructed_activation = sae(activation.to(sae.dtype)).to(dtype)

        if not reconstruct_bos_token:
            if attention_mask is not None:
                # We don't want to reconstruct at the first sequence token (<|begin_of_text|>)
                bos_token_positions: Int[Tensor, "batch_size"] = (
                    attention_mask == 0
                ).sum(dim=1)
                reconstructed_activation[:, bos_token_positions, :] = activation[
                    :, bos_token_positions, :
                ]
            elif seq_pos > 1:
                # we assume that the first token is always the <|begin_of_text|> token in case
                # the prompt contains multiple sequence positions (if seq_pos == 1 we're probably generating)
                reconstructed_activation[:, 0, :] = activation[:, 0, :]

        if isinstance(input, tuple):
            return (reconstructed_activation, *input[1:])
        else:
            return reconstructed_activation

    return hook_fn


def get_sae_hooks(
    model_block_modules: List[torch.nn.Module],
    sae_dict,
    reconstruct_bos_token: bool = False,
):
    fwd_hooks = []

    fwd_pre_hooks = [
        (
            model_block_modules[layer],
            get_sae_fwd_pre_hook(
                sae=sae_dict[layer],
                reconstruct_bos_token=reconstruct_bos_token,
            ),
        )
        for layer in range(len(model_block_modules))
        if layer in sae_dict.keys()
    ]

    return fwd_hooks, fwd_pre_hooks


def get_sae_encoded_hook(
    sae,
    cache: Float[Tensor, "batch pos d_sae"],
    positions: List[int],
):
    def hook_fn(
        module,
        input,
        output,
    ):
        nonlocal sae, cache

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        pos_activation = activation[:, positions, :]

        sae_activation = sae.encode(pos_activation.to(torch.float32)).to(cache)

        if cache.shape[0] == sae_activation.shape[0]:
            cache[:, :, :] = sae_activation
        else:
            raise ValueError(
                f"Cache batch size {cache.shape[0]} doesn't match encoded_activations batch size {sae_activation.shape[0]}"
            )

    return hook_fn


def get_reconstruction_capture_hook(
    cache: Tensor,
    sae,
    positions: List[int],
    reconstruct_bos_token: bool = False,
):
    def hook_fn(module, input):
        if isinstance(input, tuple):
            activation: Tensor = input[0].clone()
        else:
            activation: Tensor = input.clone()

        reconstructed_activation = sae(activation.to(torch.float32)).to(
            activation.dtype
        )

        if not reconstruct_bos_token:
            reconstructed_activation[:, 0, :] = activation[:, 0, :]

        cache[:, :, :] += reconstructed_activation[:, positions, :]

    return hook_fn
