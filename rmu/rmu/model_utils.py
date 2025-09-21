"""Taken from: https://github.com/centerforaisafety/wmdp/blob/main/rmu/utils.py"""

import json
import math
import random
from typing import Literal

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rmu import HF_TOKEN

random.seed(0)

################################
##### Activation functions #####
################################


def forward_with_cache(model, inputs, module, args, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0].cpu())  # load to cpu first, move to gpu later
        else:
            cache.append(output.cpu())
        return None

    hook_handle = module.register_forward_hook(hook)

    if no_grad:
        with torch.no_grad():
            _ = model(**inputs, use_cache=not args.gradient_checkpointing)
    else:
        _ = model(**inputs, use_cache=not args.gradient_checkpointing)

    hook_handle.remove()

    return cache[0]


#######################################
##### Model and data loading code #####
#######################################


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def load_model(model_name_or_path, args):
    torch_dtype = (
        "auto"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "token": HF_TOKEN,
    }

    if args.training:
        if model_name_or_path in ["google/gemma-2-2b", "google/gemma-2-2b-it"]:
            model_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False, token=HF_TOKEN
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_data(forget_corpora, retain_corpora, min_len=50, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x["text"]) > min_len:
                    data.append(str(x["text"]))
        else:
            for line in tqdm(
                open(f"data/{name}.jsonl", "r"),
                desc=f"Reading {name} data",
            ):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)["text"]
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        data = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora],
    )


def calculate_similarity(
    u: torch.Tensor, v: torch.Tensor, type: Literal["cosine", "angular"] = "cosine"
):
    sim = torch.nn.functional.cosine_similarity(u, v, dim=-1)
    if type == "angular":
        # Based on https://arxiv.org/abs/1803.11175
        # and https://math.stackexchange.com/questions/2874940/cosine-similarity-vs-angular-distance
        sim = torch.clamp(sim, -1.0, 1.0)
        sim = torch.ones_like(sim) - (torch.acos(sim) / math.pi)
    return sim.mean().item()
