import contextlib
import functools
from typing import Callable, List, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from sae_refusal import clear_memory
from sae_refusal.pipeline.hook import (
    add_hooks,
    get_activations_hook,
    get_activations_pre_hook,
)
from sae_refusal.sae.hook import get_sae_encoded_hook, get_reconstruction_capture_hook


@torch.no_grad()
def get_activations_pre(
    model,
    tokenizer,
    instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=32,
    layers=None,
    positions=[-1],
    verbose=True,
):
    torch.cuda.empty_cache()

    if layers is None:
        layers = range(model.config.num_hidden_layers)

    n_positions = len(positions)
    n_layers = len(layers)
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the activations in high-precision to avoid numerical issues
    activations = torch.zeros(
        (len(instructions), n_layers, n_positions, d_model), device=model.device
    )

    for i in tqdm(range(0, len(instructions), batch_size), disable=not verbose):
        end_idx = min(i + batch_size, len(instructions))

        inputs = tokenize_instructions_fn(instructions=instructions[i:end_idx])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        inputs_len = len(inputs["input_ids"])

        fwd_pre_hooks = [
            (
                block_modules[layer],
                get_activations_pre_hook(
                    cache=activations[i : i + inputs_len, layer_idx, :, :],
                    n_samples=n_samples,
                    positions=positions,
                ),
            )
            for layer_idx, layer in enumerate(layers)
        ]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            _ = model(**inputs)

        del inputs
        clear_memory()

    return activations


@torch.no_grad()
def get_sae_reconstructed_pre(
    model,
    tokenize_instructions_fn,
    instructions: List[str],
    sae_dict,
    block_modules,
    batch_size: int = 32,
    layers: List[int] = None,
    positions: List[int] = [-1],
    reconstruct_bos_token: bool = False,
    verbose: bool = True,
):
    torch.cuda.empty_cache()
    model.eval()

    # If no specific layers are requested, use all layers that have an SAE
    if layers is None:
        layers = sorted(list(sae_dict.keys()))

    n_positions = len(positions)
    n_layers = len(layers)
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # Initialize the cache tensor to store the results
    activations = torch.zeros(
        (n_samples, n_layers, n_positions, d_model),
        device=model.device,
        dtype=model.dtype,
    )

    for i in tqdm(
        range(0, n_samples, batch_size),
        disable=not verbose,
        desc="Getting SAE Reconstructions",
    ):
        end_idx = min(i + batch_size, n_samples)
        batch_instructions = instructions[i:end_idx]

        inputs = tokenize_instructions_fn(instructions=batch_instructions)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        current_batch_size = len(inputs["input_ids"])

        # Create a list of hooks for the current batch
        fwd_pre_hooks = []
        for layer_idx, layer in enumerate(layers):
            if layer not in sae_dict:
                print(f"Warning: Layer {layer} not in sae_dict, skipping.")
                continue

            sae = sae_dict[layer]

            # The cache slice for the current batch and layer
            cache_slice = activations[i : i + current_batch_size, layer_idx, :, :]

            hook = get_reconstruction_capture_hook(
                cache=cache_slice,
                sae=sae,
                positions=positions,
                reconstruct_bos_token=reconstruct_bos_token,
            )
            fwd_pre_hooks.append((block_modules[layer], hook))

        # Run the forward pass with the hooks attached
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            _ = model(**inputs)

    return activations


@torch.no_grad()
def get_sae_encoded_features(
    model,
    tokenizer,
    instructions,
    tokenize_instructions_fn,
    block_modules,
    saes_per_layer,  # Should be a dict of SAE per layer
    batch_size=32,
    layers=None,
    positions=None,  # e.g., [-1] for last token
    verbose=True,
):
    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    if positions is None:
        positions = [-1]  # Default to last token

    # Ensure all specified layers have a corresponding SAE
    for layer_idx in layers:
        if layer_idx not in saes_per_layer:
            raise ValueError(f"SAE not provided for layer {layer_idx}")
        # Set to eval mode
        saes_per_layer[layer_idx] = saes_per_layer[layer_idx].eval()

    n_positions = len(positions)
    n_layers = len(layers)
    n_samples = len(instructions)

    # We assume all SAE has the same d_sae
    d_sae_output = saes_per_layer[layers[0]].d_sae

    sae_features = torch.zeros(
        (n_samples, n_layers, n_positions, d_sae_output),
        dtype=torch.float64,
        device="cpu",
    )  # save memory

    for i in tqdm(
        range(0, n_samples, batch_size),
        desc="Extracting SAE Features",
        disable=not verbose,
    ):
        end_idx = min(i + batch_size, n_samples)

        inputs = tokenize_instructions_fn(instructions=instructions[i:end_idx])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        local_caches = []
        hooks_to_register = []

        for idx, actual_model_layer_idx in enumerate(layers):
            target_module = block_modules[actual_model_layer_idx]
            current_sae = saes_per_layer[actual_model_layer_idx]

            local_cache = torch.zeros(
                (len(instructions[i:end_idx]), n_positions, current_sae.d_sae),
                device=model.device,
                dtype=torch.float64,
            )
            local_caches.append(local_cache)

            hook_fn = get_sae_encoded_hook(
                sae=current_sae,
                cache=local_cache,
                positions=positions,
            )
            hooks_to_register.append((target_module, hook_fn))

        with add_hooks(
            module_forward_pre_hooks=[], module_forward_hooks=hooks_to_register
        ):
            outputs = model(**inputs)

        # Now copy from all_local_caches_for_batch to sae_features
        for idx, local_cache_filled in enumerate(local_caches):
            # sae_features shape: (n_samples, n_layers, n_positions, d_sae_output)
            # We are filling for samples [current_batch_start_idx : current_batch_end_idx]
            # and for the layer corresponding to layer_list_idx in the `layers` list.
            sae_features[
                i:end_idx,
                idx,
                :,
                :,
            ] = local_cache_filled.cpu()

        del inputs
        del outputs
        del local_caches  # Free GPU memory for these batch-specific caches
        clear_memory()

    return sae_features


@torch.no_grad()
def get_activations(
    model,
    tokenizer,
    instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=32,
    layers=None,
    positions=[-1],
    verbose=True,
):
    torch.cuda.empty_cache()

    if layers is None:
        layers = range(model.config.num_hidden_layers)

    n_positions = len(positions)
    n_layers = len(layers)
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the activations in high-precision to avoid numerical issues
    activations = torch.zeros(
        (len(instructions), n_layers, n_positions, d_model), device=model.device
    )

    for i in tqdm(range(0, len(instructions), batch_size), disable=not verbose):
        end_idx = min(i + batch_size, len(instructions))

        inputs = tokenize_instructions_fn(instructions=instructions[i:end_idx])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        inputs_len = len(inputs["input_ids"])

        fwd_hooks = [
            (
                block_modules[layer],
                get_activations_hook(
                    cache=activations[i : i + inputs_len, layer_idx, :, :],
                    positions=positions,
                ),
            )
            for layer_idx, layer in enumerate(layers)
        ]

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            _ = model(**inputs)

        del inputs
        clear_memory()

    return activations


@torch.enable_grad()
def get_refusal_gradients(
    model,
    tokenize_instructions_fn,
    prompts: List[str],
    block_modules: List[torch.nn.Module],
    refusal_direction: torch.Tensor,
    gradient_layer_idx: int,
    refusal_layer_idx: int,
    batch_size: int = 8,
    positions=[-1],
):
    model.eval()

    device = model.device
    refusal_direction = refusal_direction.to(device)

    all_gradients = []

    for i in tqdm(
        range(0, len(prompts), batch_size), desc="Calculating Refusal Gradients"
    ):
        batch_prompts = prompts[i : i + batch_size]

        captured_activations = {}

        def hook_fn(layer_idx):
            def hook(module, input):
                if isinstance(input, tuple):
                    activations = input[0]
                else:
                    activations = input
                activations.requires_grad_(True)
                captured_activations[layer_idx] = activations

            return hook

        fwd_pre_hooks = [
            (block_modules[gradient_layer_idx], hook_fn(gradient_layer_idx)),
            (block_modules[refusal_layer_idx], hook_fn(refusal_layer_idx)),
        ]

        inputs = tokenize_instructions_fn(instructions=batch_prompts)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        attention_mask = inputs["attention_mask"]

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            _ = model(**inputs)

        # Get activations for both layers
        early_activations = captured_activations[gradient_layer_idx]
        late_activations = captured_activations[refusal_layer_idx]

        # 1. Gather the hidden states at ALL specified positions for each prompt
        # Shape: [batch_size, num_positions, d_model]
        positional_late_activations = late_activations[:, positions, :]

        # 2. Project each of these activations onto the refusal direction.
        # We use einsum for a clean and efficient dot product across the batch.
        # 'bpd,d->bp' means: for batch (b), position (p), and dimension (d),
        # multiply by the direction (d) to get a result per batch and position (bp).
        projections = torch.einsum(
            "bpd,d->bp",
            positional_late_activations,
            refusal_direction.to(positional_late_activations.dtype),
        )
        # Shape of projections: [batch_size, num_positions]

        # 3. Use the attention mask to zero out projections from padding tokens.
        # This ensures short prompts are handled correctly.
        positional_attention_mask = attention_mask[:, positions]
        masked_projections = projections * positional_attention_mask.to(
            projections.dtype
        )

        # 4. Sum the valid projections to get the final scalar metric R for each sample.
        R_batch = masked_projections.sum(dim=1)
        # Shape of R_batch: [batch_size]

        # Use torch.autograd.grad for efficient per-sample gradient calculation
        # This computes the gradient of sum(R_batch) w.r.t early_activations,
        # which, because each R_i only depends on x_i, gives the per-sample gradients.
        (batch_gradients,) = torch.autograd.grad(
            outputs=R_batch,
            inputs=early_activations,
            grad_outputs=torch.ones_like(R_batch),  # Needed for vector-valued outputs
            retain_graph=False,  # We don't need the graph after this
        )

        # Append the detached gradients for this batch to our list
        # We move to CPU to free up GPU memory
        all_gradients.append(batch_gradients.sum(dim=1).detach().cpu())

        del inputs
        clear_memory()

    # Concatenate all gradient tensors from all batches into a single tensor
    final_gradients = torch.cat(all_gradients, dim=0).mean(dim=0)

    return final_gradients
