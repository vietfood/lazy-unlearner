import functools
import json
import math
import os
from typing import List, Literal, Optional

import torch
from einops import rearrange
from jaxtyping import Float, Int
from rouge_score import rouge_scorer
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from sae_refusal import clear_memory
from sae_refusal.model import ModelBase
from sae_refusal.model.embedding import EmbeddingModel
from sae_refusal.pipeline.hook import (
    add_hooks,
    get_activation_addition_input_pre_hook_rmu,
    get_activations_hook,
    get_all_direction_ablation_hooks_rmu,
)
from sae_refusal.pipeline.utils import (
    get_last_position_logits,
    kl_div_fn,
)
from sae_refusal.plot import (
    plot_refusal_scores,
    plot_refusal_scores_plotly,
)
from sae_refusal.probe import LinearProbe


def junk_score(
    activations: Float[Tensor, "batch seq d_model"],
    probe: LinearProbe,
    probe_mean: Float[Tensor, "batch d_model"],
    probe_std: Float[Tensor, "batch d_model"],
):
    assert probe_mean.dtype == probe_std.dtype

    activations = activations.to(probe_mean.dtype)

    # Take activations at the last token and normalized
    norm_activations: Float[Tensor, "batch d_model"] = (
        activations[:, -1, :] - probe_mean
    ) / (probe_std + 1e-8)

    # Score through probe
    return probe(norm_activations)


def get_junk_scores(
    model: ModelBase,
    instructions,
    probe,
    probe_mean,
    probe_std,
    junk_layer,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    batch_size=32,
    progress_bar=True,
):
    junk_score_fn = functools.partial(
        junk_score, probe=probe, probe_mean=probe_mean, probe_std=probe_std
    )

    junk_scores = torch.zeros(len(instructions), device=model.model.device)

    for i in tqdm(
        range(0, len(instructions), batch_size),
        desc="Calculating Junk Scores",
        disable=not progress_bar,
    ):
        end_idx = min(i + batch_size, len(instructions))

        tokenized_instructions = model.tokenize_instructions_fn(
            instructions=instructions[i:end_idx]
        )
        tokenized_instructions = {
            k: v.to(model.model.device) for k, v in tokenized_instructions.items()
        }

        with add_hooks(
            module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
        ):
            final_activations = model.model(
                **tokenized_instructions, output_hidden_states=True
            ).hidden_states[junk_layer]

        junk_scores[i:end_idx] = junk_score_fn(activations=final_activations)

        del tokenized_instructions
        clear_memory()

    return junk_scores


# returns True if the direction should be filtered out
def filter_fn(
    refusal_score,
    steering_score,
    kl_div_score,
    layer,
    n_layer,
    kl_threshold=None,
    induce_refusal_threshold=None,
    prune_layer_percentage=0.20,
) -> bool:
    if (
        math.isnan(refusal_score)
        or math.isnan(steering_score)
        or math.isnan(kl_div_score)
    ):
        return True
    if prune_layer_percentage is not None and layer >= int(
        n_layer * (1.0 - prune_layer_percentage)
    ):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if (
        induce_refusal_threshold is not None
        and steering_score < induce_refusal_threshold
    ):
        return True
    return False


def select_direction(
    base_model: ModelBase,
    rmu_model: ModelBase,
    instructions,
    candidate_directions: List[Float[Tensor, "n_pos n_layer d_model"]],
    probe,
    probe_mean,
    probe_std,
    junk_layer,
    artifact_dir,
    kl_threshold=0.1,
    induce_refusal_threshold=0.0,  # directions with a lower inducing refusal score are filtered out
    prune_layer_percentage=0.2,  # discard the directions extracted from the last 20% of the model
    batch_size=32,
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    assert base_model.model.device == rmu_model.model.device

    n_pos, n_layer, _ = candidate_directions[0].shape

    junks_fn = functools.partial(
        get_junk_scores,
        instructions=instructions,
        probe=probe,
        probe_mean=probe_mean,
        probe_std=probe_std,
        batch_size=batch_size,
        junk_layer=junk_layer,
    )

    baseline_junk_scores_base = junks_fn(
        model=base_model,
        fwd_hooks=[],
        fwd_pre_hooks=[],
        progress_bar=True,
    )

    baseline_junk_scores_rmu = junks_fn(
        model=rmu_model,
        fwd_hooks=[],
        fwd_pre_hooks=[],
        progress_bar=True,
    )

    ablation_kl_div_scores = torch.zeros(
        (n_pos, n_layer), device=base_model.model.device, dtype=torch.float64
    )
    ablation_junk_scores = torch.zeros(
        (n_pos, n_layer), device=base_model.model.device, dtype=torch.float64
    )
    steering_junk_scores = torch.zeros(
        (n_pos, n_layer), device=base_model.model.device, dtype=torch.float64
    )

    baseline_base_logits = get_last_position_logits(
        model=base_model.model,
        tokenizer=base_model.tokenizer,
        instructions=instructions,
        tokenize_instructions_fn=base_model.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size,
    )

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(
            range(n_layer),
            desc=f"Computing KL between Ablation and Non-Ablation for Base Model at source position {source_pos}",
        ):
            ablation_dirs = [
                dir[source_pos, source_layer] for dir in candidate_directions
            ]

            fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks_rmu(
                model_base=base_model,
                directions=ablation_dirs,
            )

            intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = (
                get_last_position_logits(
                    model=base_model.model,
                    tokenizer=base_model.tokenizer,
                    instructions=instructions,
                    tokenize_instructions_fn=base_model.tokenize_instructions_fn,
                    fwd_pre_hooks=fwd_pre_hooks,
                    fwd_hooks=fwd_hooks,
                    batch_size=batch_size,
                )
            )

            ablation_kl_div_scores[source_pos, source_layer] = (
                kl_div_fn(baseline_base_logits, intervention_logits, mask=None)
                .mean(dim=0)
                .item()
            )

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(
            range(n_layer),
            desc=f"Computing Junk Ablation on RMU Model at source position {source_pos}",
        ):
            ablation_dirs = [
                dir[source_pos, source_layer] for dir in candidate_directions
            ]

            fwd_pre_hooks, fwd_hooks = get_all_direction_ablation_hooks_rmu(
                model_base=rmu_model,
                directions=ablation_dirs,
            )

            junk_scores_rmu = junks_fn(
                model=rmu_model,
                fwd_hooks=fwd_hooks,
                fwd_pre_hooks=fwd_pre_hooks,
                progress_bar=False,
            )

            ablation_junk_scores[source_pos, source_layer] = (
                junk_scores_rmu.mean().item()
            )

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(
            range(n_layer),
            desc=f"Computing Junk Addition on Base Model at source position {source_pos}",
        ):
            addition_dirs = [
                dir[source_pos, source_layer] for dir in candidate_directions
            ]
            coeffs = [torch.tensor(1.0) for _ in candidate_directions]

            fwd_pre_hooks = [
                (
                    base_model.model_block_modules[source_layer],
                    get_activation_addition_input_pre_hook_rmu(
                        vectors=addition_dirs, coeffs=coeffs
                    ),
                )
            ]
            fwd_hooks = []

            junk_scores_base = junks_fn(
                model=base_model,
                fwd_hooks=fwd_hooks,
                fwd_pre_hooks=fwd_pre_hooks,
                progress_bar=False,
            )

            steering_junk_scores[source_pos, source_layer] = (
                junk_scores_base.mean().item()
            )

    raw_scores = {
        "base_ablation": baseline_junk_scores_rmu,
        "base_steer": baseline_junk_scores_base,
        "ablation": ablation_junk_scores,
        "steering": steering_junk_scores,
        "kl": ablation_kl_div_scores,
    }

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []

    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):
            json_output_all_scores.append(
                {
                    "position": source_pos,
                    "layer": source_layer,
                    "ablation_score": ablation_junk_scores[
                        source_pos, source_layer
                    ].item(),
                    "steering_score": steering_junk_scores[
                        source_pos, source_layer
                    ].item(),
                    "kl_div_score": ablation_kl_div_scores[
                        source_pos, source_layer
                    ].item(),
                }
            )

            ablation_scores = ablation_junk_scores[source_pos, source_layer].item()
            steering_score = steering_junk_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()

            sorting_score = -ablation_scores

            discard_direction = filter_fn(
                refusal_score=ablation_scores,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage,
            )

            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append(
                {
                    "position": source_pos,
                    "layer": source_layer,
                    "ablation_score": ablation_junk_scores[
                        source_pos, source_layer
                    ].item(),
                    "steering_score": steering_junk_scores[
                        source_pos, source_layer
                    ].item(),
                    "kl_div_score": ablation_kl_div_scores[
                        source_pos, source_layer
                    ].item(),
                }
            )

    with open(f"{artifact_dir}/direction_evaluations.json", "w") as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(
        json_output_filtered_scores, key=lambda x: x["ablation_score"], reverse=False
    )

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", "w") as f:
        json.dump(json_output_filtered_scores, f, indent=4)

    if len(filtered_scores) == 0:
        print("Warning: All scores have been filtered out!")
        return raw_scores, -1, -1, []

    # sorted in descending order
    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse=True)

    # now return the best position, layer, and direction
    _, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(
        f"Ablation score: {ablation_junk_scores[pos, layer]:.4f} (baseline: {baseline_junk_scores_rmu.mean().item():.4f})"
    )
    print(
        f"Steering score: {steering_junk_scores[pos, layer]:.4f} (baseline: {baseline_junk_scores_base.mean().item():.4f})"
    )
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")

    return raw_scores, pos, layer, [dir[pos, layer] for dir in candidate_directions]
