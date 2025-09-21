import json
import os
import time
from typing import List

import optuna
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from tqdm.auto import tqdm

from rmu.config import UnlearnConfig
from rmu.model_utils import (
    calculate_similarity,
    forward_with_cache,
    get_params,
    load_model,
)
from rmu.utils import setup_output_directory


def clear_cuda():
    # Clean up memory
    import gc

    gc.collect()

    if torch.cuda.is_available():
        with torch.no_grad():
            torch.cuda.empty_cache()


class UnlearnTrainer:
    def __init__(
        self,
        forget_data: List[str],
        retain_data: List[str],
        args: UnlearnConfig,
        trial: optuna.Trial = None,
    ):
        self.args = args
        self.forget_data_list = forget_data
        self.retain_data_list = retain_data
        self.trial = trial

        # ----- Setup Accelerator -----
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        self.device = self.accelerator.device
        set_seed(args.seed)

        # ----- Setup Model -----
        print(f"Loading models: {args.model_name_or_path}")

        self.frozen_model_base, self.tokenizer = load_model(
            self.args.model_name_or_path, args
        )
        self.updated_model_base, _ = load_model(self.args.model_name_or_path, args)

        # Enable gradient checkpointing
        if (
            hasattr(self.updated_model_base, "gradient_checkpointing_enable")
            and args.gradient_checkpointing
        ):
            print("Enabling gradient checkpointing.")

            self.updated_model_base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gc_use_reentrant}
            )

        # Set to train and eval mode
        self.updated_model_base.train()
        self.frozen_model_base.eval()

        # ----- Set up params and optimizer -----
        self.layers_ids = [
            self.args.layer_id - 2,
            self.args.layer_id - 1,
            self.args.layer_id,
        ]

        self.params_to_update = get_params(
            self.updated_model_base, self.layers_ids, args.param_ids
        )

        # Use fused = True to improve AdamW
        self.optimizer = AdamW(self.params_to_update, lr=args.lr, fused=True)

        # ----- Something else -----
        if trial is None or self.accelerator.is_main_process:
            self.accelerator.print(
                f"--- Running RMU {f'Trial Number {trial.number}' if trial else ''}---"
            )
            self.accelerator.print(
                f"Current HPs:\nLayer_id={args.layer_id}\nLR={args.lr}\nAlpha={args.alpha}\nSteering_coeffs={args.steering_coeffs}\nMax_Batches={args.max_num_batches}\nOptimizer={args.optimizer}\nAdd_Special_Tokens={args.add_special_tokens}"
            )
            self.accelerator.print(
                f"Using device: {self.device}, Processes: {self.accelerator.num_processes}, Distributed: {self.accelerator.distributed_type}, Precision: {self.accelerator.mixed_precision}"
            )
            if args.output_dir and trial is None:
                self.accelerator.print(f"Model will be saved to: {args.output_dir}")

    def __del__(self):
        del self.tokenizer
        del self.frozen_model_base
        del self.updated_model_base
        clear_cuda()

    def run(self, save: bool = False):
        # Prepare everything with Accelerator
        updated_model, optimizer, frozen_model = self.accelerator.prepare(
            self.updated_model_base,
            self.optimizer,
            self.frozen_model_base,
        )

        # Wandb config logging (only on main process)
        if self.accelerator.is_main_process and wandb.run:
            trial_hps = {
                "optuna_trial_number": self.trial.number if self.trial else -1,
            }
            current_wandb_config = self.args.dict()
            current_wandb_config.update(trial_hps)
            wandb.config.update(current_wandb_config, allow_val_change=True)

        # When using DDP, the model gets wrapped.
        unwrapped_updated_model = self.accelerator.unwrap_model(updated_model)
        unwrapped_frozen_model = self.accelerator.unwrap_model(frozen_model)

        # Access modules from the unwrapped model
        frozen_module = eval(
            self.args.module_str.format(
                model_name="unwrapped_frozen_model", layer_id=self.args.layer_id
            )
        )

        updated_module = eval(
            self.args.module_str.format(
                model_name="unwrapped_updated_model", layer_id=self.args.layer_id
            )
        )

        # Create control vector list
        control_vectors_list = []
        for i in range(len(self.forget_data_list)):
            random_vector = torch.rand(
                1,
                1,
                unwrapped_updated_model.config.hidden_size,
                dtype=unwrapped_updated_model.dtype,
                device=self.device,
            )
            control_vec = (
                random_vector / torch.norm(random_vector) * self.args.steering_coeffs[i]
            )
            control_vectors_list.append(control_vec)

        num_batches_per_epoch = self.args.max_num_batches

        # Metric to optimze: unlearn_loss (minimize), retain_loss (minimize), retain_sim (maximize)
        # Initialize first metric with bad value
        final_metric_to_optimize = (float("inf"), float("inf"), float("inf"), float("-inf"))

        if num_batches_per_epoch == 0:
            self.accelerator.print("Warning: num_batches_per_epoch is 0.")
            return final_metric_to_optimize

        if self.trial is None or self.accelerator.is_main_process:
            self.accelerator.print(
                f"Starting training for {self.args.num_epochs} epochs, {num_batches_per_epoch} batches per epoch."
            )
            self.accelerator.print(f"RMU Type: {self.args.type}")
            if self.args.type == "adaptive":
                self.accelerator.print(f"Adaptive Scale: {self.args.scale}")

        if self.args.type == "adaptive":
            coeffs = {"0": 1.0, "1": 1.0}

        truncation_side = self.tokenizer.truncation_side
        self.tokenizer.truncation_side = "right"
        global_step = 0

        for epoch in range(self.args.num_epochs):
            epoch_metrics_sum = {
                "total_loss": 0.0,
                "unlearn_loss": 0.0,
                "retain_loss": 0.0,
                "unlearn_sim": 0.0,
                "retain_sim": 0.0,
            }

            if self.accelerator.is_main_process:
                self.accelerator.print(
                    f"\n--- Epoch {epoch + 1}/{self.args.num_epochs} ---"
                )

            # Progress bar only on the main process
            pbar = tqdm(
                total=num_batches_per_epoch,
                desc=f"Epoch {epoch + 1}",
                unit="batch",
                disable=not self.accelerator.is_main_process,
            )

            for idx in range(num_batches_per_epoch):
                with self.accelerator.accumulate(updated_model):
                    batch_start_time = time.time()
                    batch_metrics = {
                        "total_loss": 0.0,
                        "unlearn_loss": 0.0,
                        "retain_loss": 0.0,
                        "unlearn_sim": 0.0,
                        "retain_sim": 0.0,
                    }

                    topic_idx = idx % len(self.forget_data_list)
                    batch_idx = idx // len(self.forget_data_list)

                    control_vec = control_vectors_list[topic_idx]
                    unlearn_batch = self.forget_data_list[topic_idx][batch_idx]
                    retain_batch = self.retain_data_list[topic_idx][batch_idx]

                    max_length = 512 if topic_idx == 0 else 768

                    # Setup unlearn inputs and loss
                    unlearn_inputs = self.tokenizer(
                        unlearn_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=max_length,
                    )
                    unlearn_inputs = {
                        k: v.to(self.device) for k, v in unlearn_inputs.items()
                    }

                    updated_forget_activations = forward_with_cache(
                        unwrapped_updated_model,
                        unlearn_inputs,
                        module=updated_module,
                        args=self.args,
                        no_grad=False,
                    ).to(self.device)

                    if self.args.type == "adaptive":
                        if idx == 0:
                            coeffs["0"] = (
                                torch.mean(
                                    updated_forget_activations.norm(dim=-1).mean(dim=1),
                                    dim=0,
                                ).item()
                                * self.args.scale
                            )
                        elif idx == 1:
                            coeffs["1"] = (
                                torch.mean(
                                    updated_forget_activations.norm(dim=-1).mean(dim=1),
                                    dim=0,
                                ).item()
                                * self.args.scale
                            )
                        else:
                            pass
                        unlearn_loss = torch.nn.functional.mse_loss(
                            updated_forget_activations,
                            control_vec.expand_as(updated_forget_activations)
                            * coeffs[f"{topic_idx}"],
                        )
                    else:
                        unlearn_loss = torch.nn.functional.mse_loss(
                            updated_forget_activations,
                            control_vec.expand_as(updated_forget_activations),
                        )

                    # Setup retain inputs and loss
                    retain_inputs = self.tokenizer(
                        retain_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=512,
                    )
                    retain_inputs = {
                        k: v.to(self.device) for k, v in retain_inputs.items()
                    }

                    updated_retain_activations = forward_with_cache(
                        unwrapped_updated_model,
                        retain_inputs,
                        module=updated_module,
                        args=self.args,
                        no_grad=False,
                    ).to(self.device)

                    frozen_retain_activations = forward_with_cache(
                        unwrapped_frozen_model,
                        retain_inputs,
                        module=frozen_module,
                        args=self.args,
                        no_grad=True,
                    ).to(self.device)

                    retain_loss = torch.nn.functional.mse_loss(
                        updated_retain_activations, frozen_retain_activations
                    )

                    # Total loss
                    loss = unlearn_loss + (self.args.alpha[topic_idx] * retain_loss)
                    self.accelerator.backward(loss)

                # It's time for optimizer to step after gradient accumulation
                if self.accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if self.accelerator.is_main_process:
                        pbar.update(1)

                        # Calculate average batch loss
                        batch_metrics["total_loss"] = (
                            self.accelerator.gather_for_metrics(loss.detach())
                            .mean()
                            .item()
                        )

                        batch_metrics["unlearn_loss"] = (
                            self.accelerator.gather_for_metrics(unlearn_loss.detach())
                            .mean()
                            .item()
                        )

                        batch_metrics["retain_loss"] = (
                            self.accelerator.gather_for_metrics(retain_loss.detach())
                            .mean()
                            .item()
                        )

                        # Accumulate for epoch averages
                        epoch_metrics_sum["total_loss"] += batch_metrics["total_loss"]
                        epoch_metrics_sum["unlearn_loss"] += batch_metrics[
                            "unlearn_loss"
                        ]
                        epoch_metrics_sum["retain_loss"] += batch_metrics["retain_loss"]

                        # Calculate similarity
                        with torch.no_grad():
                            frozen_forget_activations_sim = forward_with_cache(
                                unwrapped_frozen_model,
                                unlearn_inputs,
                                module=frozen_module,
                                args=self.args,
                                no_grad=True,
                            )

                            updated_forget_activations_sim = (
                                updated_forget_activations.clone().cpu()
                            )

                            updated_retain_activations_sim = (
                                updated_retain_activations.clone().cpu()
                            )

                            frozen_retain_activations_sim = (
                                frozen_retain_activations.clone().cpu()
                            )

                        batch_metrics["unlearn_sim"] = calculate_similarity(
                            updated_forget_activations_sim,
                            frozen_forget_activations_sim,
                        )

                        batch_metrics["retain_sim"] = calculate_similarity(
                            updated_retain_activations_sim,
                            frozen_retain_activations_sim,
                        )

                        epoch_metrics_sum["unlearn_sim"] += batch_metrics["unlearn_sim"]

                        epoch_metrics_sum["retain_sim"] += batch_metrics["retain_sim"]

                        log_metrics = {
                            **batch_metrics,
                        }

                        if self.args.type == "adaptive":
                            log_metrics[f"adaptive_coeff{topic_idx}"] = coeffs[
                                str(topic_idx)
                            ]

                        pbar.set_postfix({**log_metrics, "topic_idx": topic_idx})

                        if wandb.run:
                            log_payload = {**log_metrics}
                            log_payload.update(
                                {
                                    "epoch_num": epoch + 1,
                                    "batch_time": time.time() - batch_start_time,
                                    "global_step": global_step,
                                    f"topic{topic_idx}/updated_forget_norm": torch.mean(
                                        updated_forget_activations_sim.norm(
                                            dim=-1
                                        ).mean(dim=1),
                                        dim=0,
                                    ),
                                    f"topic{topic_idx}/frozen_forget_norm": torch.mean(
                                        frozen_forget_activations_sim.norm(dim=-1).mean(
                                            dim=1
                                        ),
                                        dim=0,
                                    ),
                                    f"topic{topic_idx}/updated_retain_norm": torch.mean(
                                        updated_retain_activations_sim.norm(
                                            dim=-1
                                        ).mean(dim=1),
                                        dim=0,
                                    ),
                                    f"topic{topic_idx}/frozen_retain_norm": torch.mean(
                                        frozen_retain_activations_sim.norm(dim=-1).mean(
                                            dim=1
                                        ),
                                        dim=0,
                                    ),
                                }
                            )
                            wandb.log(
                                log_payload,
                                step=global_step,
                            )

                        if self.args.verbose and (
                            global_step % 10 == 0 or idx == num_batches_per_epoch - 1
                        ):
                            self.accelerator.print(
                                f"\nGlobal Step {global_step} (Epoch {epoch + 1}, Batch {idx + 1}/{num_batches_per_epoch}):"
                            )
                            self.accelerator.print(
                                f"Loss: {batch_metrics['total_loss']:.4g}, Unlearn: {batch_metrics['unlearn_loss']:.4g}, Retain: {batch_metrics['retain_loss']:.4g}"
                            )
                            self.accelerator.print(
                                f"Unlearn Similarity: {batch_metrics['unlearn_sim']:.4f}, Retain Similarity: {batch_metrics['retain_sim']:.4f}"
                            )
                            if self.args.type == "adaptive":
                                self.accelerator.print(
                                    f"Adaptive Coeff (Topic {topic_idx}): {coeffs[str(topic_idx)]:.4f}"
                                )

                        # Remove data from memory
                        del updated_forget_activations
                        del frozen_retain_activations
                        del updated_retain_activations
                        del retain_inputs
                        del unlearn_inputs
                        clear_cuda()

            # End of batch training
            if self.accelerator.is_main_process:
                pbar.close()

            avg_epoch_metrics = {
                k: v / num_batches_per_epoch for k, v in epoch_metrics_sum.items()
            }

            final_metric_to_optimize = (
                avg_epoch_metrics["unlearn_loss"],
                avg_epoch_metrics["retain_loss"],
                avg_epoch_metrics["unlearn_sim"],
                avg_epoch_metrics["retain_sim"],
            )

            # Summary
            if self.accelerator.is_main_process:
                self.accelerator.print(f"""
--- Epoch {epoch + 1} Summary ---
Average Loss: {avg_epoch_metrics["total_loss"]:.4f}
Average Unlearn Loss: {avg_epoch_metrics["unlearn_loss"]:.4f}
Average Retain Loss: {avg_epoch_metrics["retain_loss"]:.4f}
Average Unlearn Similarity: {avg_epoch_metrics["unlearn_sim"]:.4f}
Average Retain Similarity: {avg_epoch_metrics["retain_sim"]:.4f}""")

            # Log epoch averages to wandb too
            if wandb.run:
                wandb.log(
                    {
                        f"epoch/{k}": v
                        for k, v in avg_epoch_metrics.items()
                        if isinstance(v, (int, float))
                    },
                    step=global_step,
                )

        # End of Training
        self.tokenizer.truncation_side = truncation_side

        # Ensure all processes are done before leaving
        self.accelerator.wait_for_everyone()

        # Delete for memory freeing
        del control_vectors_list
        clear_cuda()

        if save:
            if self.accelerator.is_main_process:
                self.accelerator.print(
                    f"Saving final model to {self.args.output_dir}..."
                )
                path = setup_output_directory(self.args)
                if path:
                    self.accelerator.print(f"Saving model to {path}...")
                    model = self.accelerator.unwrap_model(updated_model)
                    model.save_pretrained(path)
                    self.tokenizer.save_pretrained(path)
                    config_save_path = os.path.join(path, "unlearn_config.json")
                    with open(config_save_path, "w") as f:
                        json.dump({k: str(v) for k,v in self.args.dict().items()}, f, indent=4)
                    self.accelerator.print(f"Model and config saved to {path}")
                else:
                    self.accelerator.print(
                        "Output directory not set or creation failed. Model not saved."
                    )

        return final_metric_to_optimize
