import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from jaxtyping import Float
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

from sae_refusal.probe import LinearProbe


class ProbesTrainer:
    def __init__(
        self,
        hidden_layers,
        hidden_size,
        artifact_dir,
        save_dir,
        device,
        wandb_project="Probe_Training",
        type: str = "base",
    ):
        self.type = type
        self.artifact_dir = artifact_dir
        self.hidden_layers = range(0, hidden_layers)
        self.hidden_size = hidden_size
        self.device = device
        self.wandb_project = wandb_project

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.layer_stats = dict()

        print("Creating model")
        self.probes = {
            # Model on float32 type
            layer: LinearProbe(hidden_size).float().to(self.device)
            for layer in self.hidden_layers
        }

        print("Loading training and validation data...")
        self.train_loaders = self._load_data_for_split("train")
        self.val_loaders = self._load_data_for_split("val")
        self.test_loaders = self._load_data_for_split("test")
        print("Data loaded successfully.")

    def _load_data_for_split(self, split: str):
        loaders = {}
        label_path = f"{self.artifact_dir}/{split}/base_{split}.csv"
        labels_np = pd.read_csv(label_path)["label"].values
        labels = torch.tensor(labels_np, dtype=torch.float32)

        for layer in self.hidden_layers:
            feat_path = f"{self.artifact_dir}/{split}/layer_{layer}.pt"
            if not os.path.exists(feat_path):
                continue

            features = torch.load(feat_path, weights_only=True).float()

            if split == "train":
                mean = features.mean(dim=0, keepdim=True).detach()
                std = features.std(dim=0, keepdim=True).detach()
                self.layer_stats[layer] = (mean, std)

            mean, std = self.layer_stats[layer]
            features = (features - mean) / (std + 1e-8)

            dataset = TensorDataset(features, labels)

            if split == "train":  # Handle imbalance dataset
                class_counts = torch.bincount(labels.long())
                print(f"[Layer {layer}] Class balance: {class_counts.tolist()}")

                class_weights = 1.0 / class_counts.float()
                sample_weights = class_weights[labels.long()]

                sampler = WeightedRandomSampler(
                    weights=sample_weights, num_samples=len(labels), replacement=True
                )
                loaders[layer] = DataLoader(dataset, batch_size=32, sampler=sampler)
            else:
                loaders[layer] = DataLoader(dataset, batch_size=32, shuffle=False)
        return loaders

    def train(self, num_epochs=100, lr=1e-3, weight_decay=1e-2):
        for layer in self.hidden_layers:
            if layer not in self.probes:
                continue

            print(f"\n Setup WanDB for Layer {layer}")
            run = wandb.init(
                project=self.wandb_project,
                config={
                    "optimizer": "AdamW",
                    "loss": "BCELogitLoss",
                    "layer": layer,
                    "learning_rate": lr,
                    "epochs": num_epochs,
                    "weight_decay": weight_decay,
                    "hidden_size": self.hidden_size,
                },
                name=f"probe_layer_{layer}",
                reinit=True,
            )

            print(f"\n----- Training Probe for Layer {layer} -----")
            probe = self.probes[layer]
            train_loader = self.train_loaders.get(layer)
            val_loader = self.val_loaders.get(layer)

            if not train_loader or not val_loader:
                print(f"Skipping layer {layer} due to missing data.")
                continue

            optimizer = torch.optim.AdamW(
                probe.parameters(), lr=lr, weight_decay=weight_decay
            )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "max", patience=10
            )

            criterion = nn.BCEWithLogitsLoss()

            best_val_auc = -1.0

            for epoch in (pbar := tqdm(range(num_epochs), desc="Training epoch")):
                probe.train()
                total_loss = 0
                for features, labels in train_loader:
                    features, labels = features.to(self.device), labels.to(self.device)

                    logits = probe(features)
                    loss = criterion(logits, labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)

                val_metrics = self.evaluate(probe, val_loader, criterion=criterion)

                scheduler.step(val_metrics["roc_auc"])

                log_metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_f1": val_metrics["f1"],
                    "val_roc_auc": val_metrics["roc_auc"],
                }

                wandb.log(log_metrics, step=epoch)

                pbar.set_postfix(log_metrics)

                # Save the best probe based on validation AUC
                if val_metrics["roc_auc"] > best_val_auc:
                    best_val_auc = val_metrics["roc_auc"]

                    # Ensure the directory exists
                    save_dir = f"{self.save_dir.as_posix()}/models/probes"
                    os.makedirs(save_dir, exist_ok=True)

                    # Save the probe state
                    torch.save(
                        probe.state_dict(), f"{save_dir}/best_probe_layer_{layer}.pt"
                    )

                    print(
                        f"  -> New best probe saved for layer {layer} with AUC: {best_val_auc:.4f}"
                    )

            run.finish()

            # Load best state back into the probe
            best_state_dict = torch.load(
                f"{self.save_dir.as_posix()}/models/probes/best_probe_layer_{layer}.pt",
                weights_only=True,
            )
            self.probes[layer].load_state_dict(best_state_dict)

    def evaluate(
        self,
        probe: LinearProbe,
        data_loader: DataLoader,
        criterion,
    ):
        probe.eval()  # Set the model to evaluation mode
        total_loss = 0
        all_labels = []
        all_preds_scores = []

        with torch.no_grad():  # No need to calculate gradients during evaluation
            for features, labels in data_loader:
                features: Float[torch.Tensor, "batch d_model"] = features.to(
                    self.device
                )
                labels: Float[torch.Tensor, "batch"] = labels.to(self.device)

                # Calcualte loss
                logits = probe(features)
                loss = criterion(logits.squeeze(), labels)
                total_loss += loss.item()

                # Calculate score
                scores = logits.squeeze()

                all_labels.append(labels.cpu().numpy())
                all_preds_scores.append(scores.cpu().numpy())

        # Concatenate all batches
        all_labels = np.concatenate(all_labels)
        all_preds_scores = np.concatenate(all_preds_scores)

        # Get binary predictions for accuracy and confusion matrix
        all_preds_binary = (all_preds_scores > 0).astype(int)

        # Calculate metrics
        f1 = f1_score(all_labels, all_preds_binary)

        # Use try-except for roc_auc_score as it fails if only one class is present in labels
        roc_auc = roc_auc_score(all_labels, all_preds_scores)

        cm = confusion_matrix(all_labels, all_preds_binary)

        metrics = {
            "loss": total_loss / len(data_loader),
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "labels": all_labels,
            "scores": all_preds_scores,
        }
        return metrics
