from dataclasses import asdict, dataclass, fields
from typing import List

from simple_parsing import field


@dataclass
class UnlearnConfig:
    """
    Configuration class for RMU and Adaptive RMU scripts.
    """

    # Model name or path to the pre-trained model.
    model_name_or_path: str = field(
        default="HuggingFaceH4/zephyr-7b-beta", alias="model"
    )

    # String format for accessing specific modules in the model.
    module_str: str = field(default="{model_name}.model.layers[{layer_id}]")

    # Directory to save the output model. If None, a default path will be generated.
    output_dir: str = field(default="rmu_results")

    # List of corpora to retain.
    retain_corpora: List[str] = field(default_factory=lambda: ["wikitext", "wikitext"])

    # List of corpora to forget.
    forget_corpora: List[str] = field(
        default_factory=lambda: ["bio-forget-corpus", "cyber-forget-corpus"]
    )

    # Retain weights for each topic.
    alpha: List[float] = field(default_factory=lambda: [100.0, 100.0])

    # Steering vector weights for each topic.
    steering_coeffs: List[float] = field(default_factory=lambda: [20.0, 20.0])

    # Scale factor for adaptive RMU.
    scale: List[float] = field(default=lambda: [1.0, 1.0])

    # Learning rate.
    lr: float = field(default=5e-5)

    # Minimum length of data samples.
    min_len: int = field(default=0)

    # Batch size for processing data.
    batch_size: int = field(default=4)

    # Maximum number of batches to process.
    max_num_batches: int = field(default=80)

    # Number of epochs
    num_epochs: int = field(default=1)

    # Layer to unlearn.
    layer_id: int = field(default=7)

    # Parameters to update.
    param_ids: List[int] = field(default_factory=lambda: [6])

    gc_use_reentrant: bool = field(default=False)

    # Gradient checkpointing
    gradient_checkpointing: bool = field(default=True)

    # Random seed.
    seed: int = field(default=42)

    # Flag for verbose logging.
    verbose: bool = field(default=True)

    # Type of RMU method (Original or Adaptive)
    type: str = field(default="original")  # adaptive

    # Wandb
    use_wandb: bool = field(default=True)
    wandb_project: str = field(default="RMU_Unlearn")

    optimizer: str = "AdamW"  # or "muon"

    gradient_accumulation_steps: int = 16

    training: bool = True

    add_special_tokens: bool = True

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    @staticmethod
    def from_dict(dict_args):
        field_set = {f.name for f in fields(UnlearnConfig) if f.init}
        filtered_args = {k: v for k, v in dict_args.items() if k in field_set}
        return UnlearnConfig(**filtered_args)
