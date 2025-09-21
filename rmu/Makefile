.PHONY: setup
setup:
	uv sync && uv run accelerate config && uv run wandb login && uv run huggingface-cli login

IDS := 8 9 10 11 12 13 14 15
SEEDS := 42

.PHONY: hp_search
hp_search:
	@for SEED in $(SEEDS); do \
		for ID in $(IDS); do \
			uv run accelerate launch hp_search.py \
			--model_name_or_path "google/gemma-2-2b-it" \
			--batch_size 4 \
			--layer_id $$ID \
			--type "original" \
			--gradient_checkpointing \
			--gradient_accumulation_steps 1 \
			--noverbose; \
		done; \
	done