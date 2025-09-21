from abc import ABC, abstractmethod
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from sae_refusal import clear_memory
from sae_refusal.pipeline.hook import add_hooks


class ModelBase(ABC):
    def __init__(
        self,
        model_name_or_path: str,
        type: str = "base",
    ):
        self.model_name_or_path = model_name_or_path
        self.type = type

        self.tokenizer_padding_side = None

        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)

        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

    def __del__(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass

    @abstractmethod
    def _get_act_add_mod_fn(
        self, direction: Float[Tensor, "d_model"], coeff: float, layer: int
    ):
        pass

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.padding_side = self.tokenizer_padding_side
        self.tokenizer.save_pretrained(path)

    def generate_completions(
        self,
        instructions: list[str],
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=16,
        max_new_tokens=64,
        progress_bar=True,
    ):
        self.model.eval()

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        # Ensure EOS token ID is set if you want generation to stop naturally
        if self.tokenizer.eos_token_id is not None:
            generation_config.eos_token_id = self.tokenizer.eos_token_id

        completions = []

        for i in tqdm(
            range(0, len(instructions), batch_size),
            desc="Generate completions",
            disable=not progress_bar,
        ):
            end_idx = min(i + batch_size, len(instructions))
            current_batch_instructions = instructions[i:end_idx]

            tokenized_instructions = self.tokenize_instructions_fn(
                instructions=current_batch_instructions
            )

            # Move to CUDA
            tokenized_instructions = {
                k: v.to(self.model.device) for k, v in tokenized_instructions.items()
            }

            with (
                add_hooks(
                    module_forward_pre_hooks=fwd_pre_hooks, 
                    module_forward_hooks=fwd_hooks
                ), 
                torch.no_grad()
            ):
                generation_toks = self.model.generate(
                    **tokenized_instructions,
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[
                    :, tokenized_instructions["input_ids"].shape[-1] :
                ]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append(
                        {
                            "prompt": instructions[i + generation_idx],
                            "response": self.tokenizer.decode(
                                generation, skip_special_tokens=True
                            ).strip(),
                        }
                    )

            # Clean up
            del tokenized_instructions, generation_toks
            clear_memory()

        return completions
