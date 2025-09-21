import json
import os
from collections import Counter
from typing import List, Optional, Tuple, Union

import einops
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from sae_refusal import clear_memory
from sae_refusal.pipeline.hook import add_hooks


def get_orthogonalized_matrix(
    matrix: Float[Tensor, "... d_model"], vec: Float[Tensor, "d_model"]
) -> Float[Tensor, "... d_model"]:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)

    proj = (
        einops.einsum(
            matrix, vec.unsqueeze(-1), "... d_model, d_model single -> ... single"
        )
        * vec
    )
    return matrix - proj


def cosine_sim(
    output_embedding: Float[Tensor, "batch d_embed"],
    input_embedding: Float[Tensor, "batch d_embed"],
):
    return torch.nn.functional.cosine_similarity(
        output_embedding, input_embedding, dim=-1
    )


def get_last_position_logits(
    model,
    tokenizer,
    instructions,
    tokenize_instructions_fn,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    batch_size=32,
) -> Float[Tensor, "n_instructions d_vocab"]:
    last_position_logits = None

    for i in range(0, len(instructions), batch_size):
        end_idx = min(i + batch_size, len(instructions))

        tokenized_instructions = tokenize_instructions_fn(
            instructions=instructions[i:end_idx]
        )
        tokenized_instructions = {
            k: v.to(model.device) for k, v in tokenized_instructions.items()
        }

        with add_hooks(
            module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
        ):
            logits = model(**tokenized_instructions).logits

        if last_position_logits is None:
            last_position_logits = logits[:, -1, :]
        else:
            last_position_logits = torch.cat(
                (last_position_logits, logits[:, -1, :]), dim=0
            )

        del tokenized_instructions
        clear_memory()

    return last_position_logits


def masked_mean(seq, mask=None, dim=1, keepdim=False):
    if mask is None:
        return seq.mean(dim=dim)

    if seq.ndim == 3:
        mask = rearrange(mask, "b n -> b n 1")

    masked_seq = seq.masked_fill(~mask, 0.0)
    numer = masked_seq.sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim)

    masked_mean = numer / denom.clamp(min=1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.0)
    return masked_mean


def kl_div_fn(
    logits_a: Float[Tensor, "batch seq_pos d_vocab"],
    logits_b: Float[Tensor, "batch seq_pos d_vocab"],
    mask: Int[Tensor, "batch seq_pos"] = None,
    epsilon: Float = 1e-6,
) -> Float[Tensor, "batch"]:
    """
    Compute the KL divergence loss between two tensors of logits.
    """
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(
        probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1
    )

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)


def generate_and_save_completions(
    model_base,
    fwd_pre_hooks,
    fwd_hooks,
    dataset,
    batch_size=16,
    filter_fn=None,
    dataset_name=None,
    artifact_dir=None,
    intervention_label=None,
    max_new_tokens=100,
):
    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    if artifact_dir is not None:
        if intervention_label is None:
            output_name = f"{artifact_dir}/{dataset_name}_completions.json"
        else:
            output_name = (
                f"{artifact_dir}/{dataset_name}_{intervention_label}_completions.json"
            )

        with open(output_name, "w") as f:
            json.dump(completions, f, indent=4)

    if filter_fn is not None:
        return [filter_fn(completion) for completion in completions]
    else:
        return completions


def proj(a, b):
    a = a.to(b.device)  # Ensure 'a' is on the same device as 'b'

    # Calculate the dot product (a ⋅ b)
    # Result shape: (..., 1)
    dot_product_ab = einops.einsum(a, b, "... d_model, ... d_model -> ...").unsqueeze(
        -1
    )  # Add a trailing dimension of 1 for broadcasting

    # Calculate the squared L2 norm of b (||b||²)
    # Result shape: (..., 1)
    # Using a small epsilon for numerical stability in case ||b|| is zero or very small
    b_squared_norm = (
        einops.einsum(b, b, "... d_model, ... d_model -> ...").unsqueeze(-1) + 1e-6
    )  # Add epsilon here

    # Calculate the projection: ((a ⋅ b) / ||b||²) * b
    # The (..., 1) shapes of dot_product_ab and b_squared_norm will broadcast correctly with b (..., d_model)
    projection = (dot_product_ab / b_squared_norm) * b
    return projection


def compute_pca(tensors: List[torch.Tensor], k=2):
    # shape: (len(tensors) * seq_len, d_model)
    X = torch.cat(tensors, dim=0)
    X_centered = X - X.mean(dim=0, keepdim=True)

    # U: (len(tensors) * seq_len, k)  left singular vectors × singular values
    # S: (k,)     singular values
    # V: (d_model, k)  right singular vectors (principal axes)
    U, S, V = torch.pca_lowrank(X_centered, center=False, q=k)

    components = V.t()  # shape (k, d_model)
    scores = U

    T, seq_len = len(tensors), tensors[0].size(0)
    scores = scores.view(T, seq_len, k)

    explained_variance = S.pow(2) / (X_centered.size(0) - 1)
    total_variance = X_centered.pow(2).sum(dim=0) / (X_centered.size(0) - 1)
    explained_ratio = explained_variance / total_variance.sum()

    return {
        "components": components,
        "scores": scores,
        "explained_variance": explained_variance,
        "explained_ratio": explained_ratio,
        "total_variance": total_variance,
    }


def _calculate_conditional_losses(
    model,
    tokenize_fn,
    instructions,
    outputs,
    fwd_pre_hooks=[],
    fwd_hooks=[],
):
    inputs = tokenize_fn(instructions=instructions, outputs=outputs)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompts_only_tokens = tokenize_fn(instructions=instructions, outputs=None)[
        "input_ids"
    ]
    prompt_lengths = [len(p) for p in prompts_only_tokens]

    labels = inputs["input_ids"].clone()
    for j, length in enumerate(prompt_lengths):
        labels[j, :length] = -100

    with (
        add_hooks(
            module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks
        ),
        torch.no_grad(),
    ):
        model_outputs = model(**inputs)
        logits = model_outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Use permute for CrossEntropyLoss shape: (B, V, T-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        token_losses = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)

        # Create a mask for the valid (non-ignored) tokens
        mask = (shift_labels != -100).float()

        # Calculate the mean loss per example (sum of token losses / num choice tokens)
        # This is the average negative log-likelihood per token.
        loss_per_example = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)

    del inputs
    del labels
    del model_outputs
    clear_memory()

    return loss_per_example


def evaluate_multiple_choice(
    model,
    questions,
    choices,
    ground_truth_indices,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    batch_size=8,
    progress_bar=True,
):
    num_questions = len(questions)

    all_instructions = []
    all_outputs = []
    num_choices_per_question = [len(c) for c in choices]

    for i, question in enumerate(questions):
        num_c = num_choices_per_question[i]
        all_instructions.extend([question] * num_c)
        all_outputs.extend(choices[i])

    total_pairs = len(all_instructions)
    all_losses = []

    for i in tqdm(
        range(0, total_pairs, batch_size),
        desc="Evaluating MCQA Pairs",
        disable=not progress_bar,
    ):
        start_idx = i
        end_idx = min(start_idx + batch_size, total_pairs)

        batch_instructions = all_instructions[start_idx:end_idx]
        batch_outputs = all_outputs[start_idx:end_idx]

        # Calculate loss for the current batch of pairs
        batch_losses = _calculate_conditional_losses(
            model.model,
            model.tokenize_instructions_fn,
            instructions=batch_instructions,
            outputs=batch_outputs,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
        )
        all_losses.extend(batch_losses.cpu().tolist())

    is_correct_list = []
    current_loss_idx = 0
    for i in range(num_questions):
        # Get the number of choices for the current original question
        num_c = num_choices_per_question[i]

        # Slice the losses that correspond to this question's choices
        question_losses = all_losses[current_loss_idx : current_loss_idx + num_c]

        # The model's prediction is the choice with the minimum loss
        prediction_idx = torch.argmin(torch.tensor(question_losses)).item()
        ground_truth_idx = ground_truth_indices[i]

        is_correct_list.append(1 if prediction_idx == ground_truth_idx else 0)

        # Move the pointer to the start of the next question's losses
        current_loss_idx += num_c

    # Calculate final accuracy
    accuracy = sum(is_correct_list) / num_questions if num_questions > 0 else 0

    return accuracy, is_correct_list
