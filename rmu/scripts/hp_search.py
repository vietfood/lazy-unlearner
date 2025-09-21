import json
from functools import partial
from pathlib import Path
from typing import List

import optuna
import wandb
from rmu.config import UnlearnConfig
from rmu.model_utils import get_data
from rmu.unlearn import UnlearnTrainer, clear_cuda
from simple_parsing import parse

OPTUNA_CONFIG = {"n_trials": 9}
SEARCH_VERSION = 5


def objective(
    trial: optuna.Trial,
    base_args: UnlearnConfig,
    forget_data_list: List[str],
    retain_data_list: List[str],
):
    trial_args = UnlearnConfig.from_dict(base_args.dict())

    # ----- Setup Hyperparemeters -----
    trial_args.lr = trial.suggest_float("lr", 5e-5, 1)

    trial_args.max_num_batches = trial.suggest_int("max_num_batches", 100, 1000)

    alpha_topic = trial.suggest_float("alpha", 100.0, 2000.0, log=True)
    trial_args.alpha = [alpha_topic, alpha_topic]

    steering_coeffs_topic = trial.suggest_float("steering_coeffs", 5.0, 300.0, log=True)
    trial_args.steering_coeffs = [steering_coeffs_topic, steering_coeffs_topic]

    # ----- Setup Wandb ------
    wandb_group_name = f"Optuna-{base_args.layer_id}-{base_args.model_name_or_path.split('/')[-1]}"
    wandb_run_name = f"Trial{trial.number}/layer{trial_args.layer_id}_lr({trial_args.lr})_batch({trial_args.max_num_batches})_alpha({alpha_topic})_steer({steering_coeffs_topic}_optimizer({trial_args.optimizer}))"

    if base_args.wandb_project:
        wandb.init(
            project=f"{base_args.wandb_project}_Search_{SEARCH_VERSION}",
            name=wandb_run_name,
            group=wandb_group_name,
            config=trial_args.dict(),
            reinit=True,
        )

    # Default metric
    metric = float("inf"), float("inf"), float("inf"), float("-inf")

    trainer = UnlearnTrainer(
        forget_data=forget_data_list,
        retain_data=retain_data_list,
        args=trial_args,
        trial=trial,
    )

    try:
        metric = trainer.run()
        print(f"Metric in trial {trial.number}: {metric}")
        return metric
    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        if wandb.run:
            wandb.finish(exit_code=1)
        return metric
    finally:
        del trainer
        clear_cuda()
        # Finish wandb run if it's still active and wasn't finished by specific handlers
        if wandb.run:
            wandb.finish(exit_code=0)


def search(args: UnlearnConfig):
    OPTUNA_DIR = Path(f"optuna_result_{SEARCH_VERSION}/layer_{args.layer_id}")
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Setup Optuna -----
    study_name = f"rmu-search-{args.model_name_or_path.split('/')[-1]}"
    storage_name = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend((OPTUNA_DIR / Path(f"{study_name}.log")).as_posix())
    )

    # Define search space
    search_space = {
        "lr": [2e-4],
        "max_num_batches": [150],
        "alpha": [300, 600, 900],
        "steering_coeffs": [350, 500, 750],
    }
    sampler = optuna.samplers.GridSampler(search_space)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        directions=[
            "minimize",
            "minimize",
            "minimize",
            "maximize",
        ],
        sampler=sampler,
    )

    # ----- Load data -----
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.batch_size,
    )

    # ----- Create objective function -----
    objective_fn = partial(
        objective,
        base_args=args,
        forget_data_list=forget_data_list,
        retain_data_list=retain_data_list,
    )

    study.optimize(
        objective_fn,
        n_trials=OPTUNA_CONFIG["n_trials"],
        timeout=None,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    print("\nOptuna Study Summary:")
    print(f"  Number of finished trials: {len(study.trials)}")

    if not study.best_trials:
        print(
            "No successful trials or no Pareto optimal solutions found. Cannot retrain/save best models."
        )
    else:
        print(
            f"\n--- Retraining and Saving Models from Pareto Front ({len(study.best_trials)} models) ---"
        )

    best_trials_info = []
    print(
        f"\n--- Saving Information for Pareto Front Trials ({len(study.best_trials)} trials) ---"
    )
    for trial_obj in study.best_trials:
        trial_info = {
            "number": trial_obj.number,
            "values": trial_obj.values,
            "params": trial_obj.params,
            "state": str(trial_obj.state),
        }
        best_trials_info.append(trial_info)

    output_file_path = OPTUNA_DIR / Path(f"best_trials_layer{args.layer_id}.json")
    try:
        with open(
            output_file_path, "w", encoding="utf-8"
        ) as f:  # Use "w" to write (or overwrite)
            json.dump(best_trials_info, f, indent=4)
        print(
            f"Successfully saved information for {len(best_trials_info)} best trials to {output_file_path}"
        )
    except Exception as e:
        print(f"Error saving best trials information: {e}")


if __name__ == "__main__":
    args = parse(UnlearnConfig)
    search(args)
