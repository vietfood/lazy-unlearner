import os
from typing import List

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from sae_refusal.model import ModelBase
from sae_refusal.pipeline.hook import add_hooks
from sae_refusal import clear_memory

def get_mean_activations_pre_hook(
    layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]
):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = (
            input[0].clone().to(cache)
        )
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)

    return hook_fn


def get_mean_activations(
    model,
    tokenizer,
    instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=32,
    positions=[-1],
    progress_bar=True,
):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros(
        (n_positions, n_layers, d_model), dtype=torch.float64, device=model.device
    )

    fwd_pre_hooks = [
        (
            block_modules[layer],
            get_mean_activations_pre_hook(
                layer=layer,
                cache=mean_activations,
                n_samples=n_samples,
                positions=positions,
            ),
        )
        for layer in range(n_layers)
    ]

    for i in tqdm(range(0, len(instructions), batch_size), desc="Calculating mean", disable=not progress_bar):
        end_idx = min(i + batch_size, len(instructions))

        inputs = tokenize_instructions_fn(instructions=instructions[i : end_idx])
        inputs = {k: v.to(model.device) for k,v in inputs.items()}

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            _ = model(**inputs)

        del inputs
        clear_memory()

    return mean_activations


def get_mean_diff_rmu(
    base_model: ModelBase,
    rmu_model: ModelBase,
    instructions,
    batch_size=32,
    positions=[-1],
):
    mean_activations_harmful = get_mean_activations(
        rmu_model.model,
        rmu_model.tokenizer,
        instructions,
        rmu_model.tokenize_instructions_fn,
        rmu_model.model_block_modules,
        batch_size=batch_size,
        positions=positions,
    )

    mean_activations_harmless = get_mean_activations(
        base_model.model,
        base_model.tokenizer,
        instructions,
        base_model.tokenize_instructions_fn,
        base_model.model_block_modules,
        batch_size=batch_size,
        positions=positions,
    )

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = (
        mean_activations_harmful - mean_activations_harmless
    )

    return mean_diff


def get_mean_diff(
    model,
    tokenizer,
    harmful_instructions,
    harmless_instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=32,
    positions=[-1],
):
    mean_activations_harmful = get_mean_activations(
        model,
        tokenizer,
        harmful_instructions,
        tokenize_instructions_fn,
        block_modules,
        batch_size=batch_size,
        positions=positions,
    )
    mean_activations_harmless = get_mean_activations(
        model,
        tokenizer,
        harmless_instructions,
        tokenize_instructions_fn,
        block_modules,
        batch_size=batch_size,
        positions=positions,
    )

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = (
        mean_activations_harmful - mean_activations_harmless
    )

    return mean_diff


def generate_directions_rmu(
    model_base: ModelBase,
    model_rmu: ModelBase,
    instructions,
    artifact_dir=None,
    artifact_name=None,
    batch_size=32,
    positions=None,
):
    assert model_base.type == model_rmu.type

    mean_diffs = get_mean_diff_rmu(
        model_base,
        model_rmu,
        instructions,
        batch_size=batch_size,
        positions=list(range(-len(model_base.eoi_toks), 0))
        if positions is None
        else positions,
    )

    if model_base.type != "none":
        assert mean_diffs.shape == (
            len(model_base.eoi_toks),
            model_base.model.config.num_hidden_layers,
            model_base.model.config.hidden_size,
        )
    assert not mean_diffs.isnan().any()

    if artifact_dir is not None:
        if not os.path.exists(artifact_dir):
            os.makedirs(artifact_dir)
        torch.save(mean_diffs, f"{artifact_dir}/{artifact_name}.pt")

    return mean_diffs


def generate_directions(
    model_base: ModelBase, harmful_instructions, harmless_instructions, artifact_dir
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    mean_diffs = get_mean_diff(
        model_base.model,
        model_base.tokenizer,
        harmful_instructions,
        harmless_instructions,
        model_base.tokenize_instructions_fn,
        model_base.model_block_modules,
        positions=list(range(-len(model_base.eoi_toks), 0)),
    )

    assert mean_diffs.shape == (
        len(model_base.eoi_toks),
        model_base.model.config.num_hidden_layers,
        model_base.model.config.hidden_size,
    )
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs
