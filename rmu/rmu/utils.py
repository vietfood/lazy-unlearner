import random
from pathlib import Path

import numpy as np
import torch
import wandb

from rmu.config import UnlearnConfig


def setup_output_directory(config: UnlearnConfig) -> Path:
    run_specific_name = (
        f"layer{config.layer_id}_alpha{config.alpha[0]}_steer{config.steering_coeffs[0]}"
    )
    run_output_dir = Path(config.output_dir) / Path(run_specific_name)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run outputs will be saved to: {run_output_dir}")
    return run_output_dir

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_wandb(config: UnlearnConfig, project_name: str | None = None):
    if config.use_wandb:
        run_output_dir = setup_output_directory(config)
        wandb_run_name = run_output_dir.name
        wandb.init(
            project=f"{config.wandb_project}" if project_name is None else project_name,
            name=wandb_run_name,
            group=f"layer_{config.layer_id}",
            config=config.dict(),
        )
        wandb.config.update({"run_output_dir": str(run_output_dir)})
    else:
        print("W&B logging disabled.")


def cleanup_wandb():
    if wandb.run is not None:
        wandb.finish()
