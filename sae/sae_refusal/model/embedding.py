import copy
import functools
from typing import List, Literal

import torch
from jaxtyping import Float
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from sae_refusal import clear_memory
from sae_refusal.model import ModelBase


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class EmbeddingModel(ModelBase):
    def _load_model(self, model_path):
        model = SentenceTransformer(model_path, device="cuda", trust_remote_code=True)
        model.eval()
        model.requires_grad_(False)

        return model

    def embed(self, instructions, **kwargs):
        return self.model.encode(instructions, **kwargs)

    def similarity(self, inputs, outputs, **kwargs):
        return self.model.similarity(inputs, outputs, **kwargs)

    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        return None

    def _get_tokenize_instructions_fn(self):
        return None

    def _get_eoi_toks(self):
        return None

    def _get_refusal_toks(self):
        return None

    def _get_model_block_modules(self):
        return None

    def _get_attn_modules(self):
        return None

    def _get_mlp_modules(self):
        return None

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return None

    def _get_act_add_mod_fn(
        self, direction: Float[Tensor, "d_model"], coeff: float, layer: int
    ):
        return None
