# Lazy Unlearner: Code & Results

This repository contains code and results accompanying the article **“Finding and Fighting the Lazy Unlearner: An Adversarial Approach”**. Our goal is to enable reproducibility of the main experiments and analyses presented in the article.

---

## Repository Structure

* `rmu/` – Code for RMU training and producing model outputs.
* `sae/` – Code for analyzing RMU models using Sparse Autoencoders (SAE) and related probes.
* `notebooks/` – Example notebooks for analysis.

> Each folder has its own environment setup (`pyproject.toml`) and scripts.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/vietfood/lazy-unlearer.git
cd lazy-unlearner
```

(Optional) On a fresh Ubuntu machine, run:

```bash
source setup.sh
```

This will update packages, install the latest CUDA, `uv`, `nvtop`, and other dependencies needed for GPU usage.

---

## 1. RMU Training

The `rmu/` folder contains scripts for RMU training. To reproduce the results:

1. **Setup the environment**:

```bash
cd rmu
make setup
```

> This installs all dependencies via `uv`. You will need valid Hugging Face and Weights & Biases (wandb) keys.

2. **Run the training script**:

* Move `run_test.py` and `run_base.sh` outside the `scripts/` folder.
* Execute the main run script:

```bash
source run_base.sh
```

This produces the RMU-trained models used in the experiments.

---

## 2. RMU Analysis

The `sae/` folder contains code for analyzing RMU models. To reproduce the analysis:

1. **Setup the environment**:

```bash
cd sae
make setup
```

> Same as above, requires Hugging Face and wandb keys.

2. **Run analysis notebooks**:

* Move the desired notebook from `notebooks/` to the main folder.
* Open and run it in your preferred environment (e.g., Jupyter or VSCode).

This will reproduce key figures and metrics from the article.

---

## Citing This Work

If you use this work in your research, please cite:

```bibtex
@misc{ln2025rmu,
    author={Nguyen Le},
    title={Finding and Fighting the Lazy Unlearner: An Adversarial Approach},
    year={2025},
    url={https://lenguyen.vercel.app/note/rmu-improv}
}
```

---

**Optional Notes / Tips**:

* Ensure your GPU drivers and CUDA toolkit are compatible with the version specified in `pyproject.toml`.
* For large models, consider using `nvtop` to monitor GPU memory usage.
* Keep `run_test.py` outside of `scripts/` folder (for RMU) and notebooks outside of `notebooks/` (for analysis) to avoid path issues.