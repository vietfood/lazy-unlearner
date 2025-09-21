import copy
import functools
from typing import List, Literal

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_refusal.model import ModelBase
from sae_refusal.pipeline.utils import get_orthogonalized_matrix

GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

GEMMA_BASE_TEMPLATE = """<start_of_turn>user:
{instruction}<end_of_turn>
<start_of_turn>assistant:
"""

GEMMA_REFUSAL_TOKS = [235285, 1718, 651, 590, 2169]


def format_instruction_gemma_chat(
    instruction: str,
    type: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if system is not None:
        raise ValueError("System prompts are not supported for Gemma models.")
    else:
        if type == "base":
            formatted_instruction = GEMMA_BASE_TEMPLATE.format(instruction=instruction)
        elif type == "instruction":
            formatted_instruction = GEMMA_CHAT_TEMPLATE.format(instruction=instruction)
        elif type == "none":
            formatted_instruction = f"{instruction} "  # trailing space at the end

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_gemma_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    type: str,
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace=True,
    max_length: int = None,
    padding: str | bool = True,
    add_special_tokens: bool = True,
):
    if outputs is not None:
        prompts = [
            format_instruction_gemma_chat(
                instruction=instruction,
                type=type,
                output=output,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_gemma_chat(
                instruction=instruction,
                type=type,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=padding,
        truncation=False,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )

    return result


def orthogonalize_gemma_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T


def act_add_gemma_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer - 1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer - 1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class GemmaModel(ModelBase):
    def _load_model(self, model_path):
        torch_dtype = (
            "auto"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cuda",
            attn_implementation="eager",
        ).eval()

        model.requires_grad_(False)

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_gemma_chat,
            tokenizer=self.tokenizer,
            type=self.type,
            system=None,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        if self.type == "instruction":
            return self.tokenizer.encode(
                GEMMA_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False
            )
        elif self.type == "base":
            return self.tokenizer.encode(
                GEMMA_BASE_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False
            )
        else:  # type = none
            return [-1]  # take the last token

    def _get_refusal_toks(self):
        return GEMMA_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block_module.self_attn for block_module in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block_module.mlp for block_module in self.model_block_modules]
        )

    def mod_model(
        self,
        direction: Float[Tensor, "d_model"],
        type: Literal["add", "ablation"] = "ablation",
        **kwargs,
    ):
        if type == "ablation":
            orthogonalize_gemma_weights(self.model, direction=direction)
        else:
            act_add_gemma_weights(self.model, direction=direction, **kwargs)

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_gemma_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_gemma_weights, direction=direction, coeff=coeff, layer=layer
        )
