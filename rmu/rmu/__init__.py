import inspect
import os

import dotenv
import torch

dotenv.load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Taken from: https://github.com/unslothai/unsloth/blob/main/unsloth/__init__.py

# Hugging Face Hub faster downloads
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# XET is slower in Colab - investigate why
keynames = "\n" + "\n".join(os.environ.keys())
if "HF_XET_HIGH_PERFORMANCE" not in os.environ:
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
if "\nCOLAB_" in keynames:
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"


def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE_TYPE: str = get_device_type()

# Reduce VRAM usage by reducing fragmentation
# And optimize pinning of memory
if DEVICE_TYPE == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
    )

# Torch 2.4 has including_emulation
if DEVICE_TYPE == "cuda":
    major_version, minor_version = torch.cuda.get_device_capability()
    SUPPORTS_BFLOAT16 = major_version >= 8

    old_is_bf16_supported = torch.cuda.is_bf16_supported
    if "including_emulation" in str(inspect.signature(old_is_bf16_supported)):

        def is_bf16_supported(including_emulation=False):
            return old_is_bf16_supported(including_emulation)

        torch.cuda.is_bf16_supported = is_bf16_supported
    else:

        def is_bf16_supported():
            return SUPPORTS_BFLOAT16

        torch.cuda.is_bf16_supported = is_bf16_supported
