.PHONY: add
add:
	uv add -r requirements.txt && uv add jupyterlab

.PHONY: setup
setup:
	uv sync && uv run accelerate config && uv run wandb login && uv run huggingface-cli login