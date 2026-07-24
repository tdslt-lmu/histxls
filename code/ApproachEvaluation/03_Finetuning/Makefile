.PHONY: help all-full philta-debug philta-full qwen3-full qwen3-0.6b-debug qwen3-0.6b-full qwen3-0.6b-hp-search qwen3-0.6b-hp-search-small qwen2.5-72b-debug qwen2.5-72b-full llama-full mt5Small-full mt5Small-debug madlad-debug madlad-full eval-philta clean

# Default target
help:
	@echo "Available targets:"
	@echo "  all-full          - Run ALL full training experiments (all 9 models)"
	@echo "  philta-debug      - Quick test run with PhilTa (10% data, 1 epoch, 2h)"
	@echo "  philta-full       - Full PhilTa training (100% data, 3 epochs, 6h)"
	@echo "  qwen3-debug       - Quick test run with Qwen3-8B (10% data, 1 epoch, 2h, 80GB GPU)"
	@echo "  qwen3-full        - Full Qwen3-8B training (80GB GPU, 8h)"
	@echo "  qwen3-0.6b-debug  - Quick test with Qwen3-0.6B (10% data, 1 epoch, 1h, 40GB GPU)"
	@echo "  qwen3-0.6b-full   - Full Qwen3-0.6B training (100% data, 10 epochs, 6h, 40GB GPU)"
	@echo "  qwen3-0.6b-hp-search       - Hyperparameter search for Qwen3-0.6B (20 trials)"
	@echo "  qwen3-0.6b-hp-search-small - Quick HP search for Qwen3-0.6B (3 trials)"
	@echo "  qwen2.5-72b-debug - Quick test with Qwen2.5-72B (10% data, 1 epoch, 2h, 2x 80GB GPU)"
	@echo "  qwen2.5-72b-full  - Full Qwen2.5-72B training (2x 80GB GPU, 32h)"
	@echo "  llama-full        - Full Llama-3.3-70B training (80GB GPU, 12h)"
	@echo "  mt5Small-debug    - Quick test with mT5-small (300M, 512 tokens, 2h, 40GB GPU)"
	@echo "  mt5Small-full     - Full mT5-small training (300M, 512 tokens, 8h, 40GB GPU)"
	@echo "  mt5Base-debug     - Quick test with mT5-base (580M, 512 tokens, 3h, 40GB GPU)"
	@echo "  mt5Base-full      - Full mT5-base training (580M, 512 tokens, 12h, 40GB GPU)"
	@echo "  longT5-debug      - Quick test with LongT5 (4K tokens, 3h, 40GB GPU)"
	@echo "  longT5-full       - Full LongT5 training (4K tokens, 16h, 40GB GPU)"
	@echo "  madlad-debug      - Quick test with MADLAD400-3B (1024 tokens, 2h, 80GB GPU)"
	@echo "  madlad-full       - Full MADLAD400-3B training (1024 tokens, 10h, 80GB GPU)"
	@echo "  eval-philta       - Evaluate finetuned PhilTa model"
	@echo "  clean             - Remove all model checkpoints and logs"

# Run all full training experiments
all-full: philta-full qwen3-full qwen2.5-72b-full llama-full mt5Small-full mt5Base-full longT5-full madlad-full
	@echo "✓ All full training runs submitted!"

# PhilTa - Debug run (10% data, 1 epoch)
philta-debug:
	python end2end_finetuning.py \
		--model_id bowphs/PhilTa \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/philta-debug \
		--epochs 1 \
		--batch_size 4 \
		--debug \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--run_name philta-debug

# PhilTa - Full training (optimized hyperparameters)
philta-full:
	python end2end_finetuning.py \
		--model_id bowphs/PhilTa \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/philta-finetuned \
		--epochs 30 \
		--batch_size 2 \
		--gradient_accumulation_steps 4 \
		--learning_rate 2e-4 \
		--warmup_steps 10 \
		--weight_decay 0.1 \
		--max_grad_norm 0.5 \
		--label_smoothing 0.1 \
		--use_expert_rules \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 15:00:00 \
		--run_name philta-end2end-full-optimized

# Qwen3-8B - Full training (requires 80GB GPU with QLoRA, optimized hyperparameters)
qwen3-full:
	python end2end_finetuning.py \
		--model_id Qwen/Qwen3-8B \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/qwen3-8b-qlora-finetuned \
		--epochs 10 \
		--batch_size 2 \
		--gradient_accumulation_steps 4 \
		--learning_rate 2.5e-4 \
		--warmup_steps 10 \
		--weight_decay 0.1 \
		--max_grad_norm 0.5 \
		--label_smoothing 0.1 \
		--use_expert_rules \
		--use_qlora \
		--lora_r 16 \
		--lora_alpha 64 \
		--lora_dropout 0.05 \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 16:00:00 \
		--run_name qwen3-8b-qlora-full-optimized

# Qwen3-8B - Debug run with QLoRA
qwen3-debug:
	python end2end_finetuning.py \
		--model_id Qwen/Qwen3-8B \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/qwen3-8b-qlora-debug \
		--epochs 1 \
		--batch_size 4 \
		--gradient_accumulation_steps 2 \
		--learning_rate 2e-4 \
		--warmup_steps 5 \
		--use_qlora \
		--lora_r 8 \
		--lora_alpha 16 \
		--debug \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--run_name qwen3-8b-qlora-debug

# Qwen3-8B - Debug run with QLoRA
qwen3-debug-test:
	python end2end_finetuning.py \
		--model_id Qwen/Qwen3-8B \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/qwen3-8b-qlora-debug-test \
		--epochs 1 \
		--batch_size 4 \
		--gradient_accumulation_steps 2 \
		--learning_rate 2e-4 \
		--warmup_steps 5 \
		--use_qlora \
		--lora_r 8 \
		--lora_alpha 16 \
		--debug \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--run_name qwen3-8b-qlora-debug-test \
		--use_expert_rules


# Qwen3-0.6B - Debug run with QLoRA (small model, fits on 40GB GPU)
qwen3-0.6b-debug:
	python end2end_finetuning.py \
		--model_id Qwen/Qwen3-0.6B \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/qwen3-0.6b-qlora-debug \
		--epochs 1 \
		--batch_size 8 \
		--gradient_accumulation_steps 2 \
		--learning_rate 2e-4 \
		--warmup_steps 5 \
		--use_qlora \
		--lora_r 16 \
		--lora_alpha 32 \
		--use_expert_rules \
		--debug \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--run_name qwen3-0.6b-qlora-debug

# Qwen3-0.6B - Full training with QLoRA (optimized hyperparameters)
qwen3-0.6b-full:
	python end2end_finetuning.py \
		--model_id Qwen/Qwen3-0.6B \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/qwen3-0.6b-qlora-finetuned \
		--epochs 10 \
		--batch_size 2 \
		--gradient_accumulation_steps 4 \
		--learning_rate 2.5e-4 \
		--warmup_steps 10 \
		--weight_decay 0.1 \
		--max_grad_norm 0.5 \
		--label_smoothing 0.1 \
		--use_qlora \
		--lora_r 16 \
		--lora_alpha 64 \
		--lora_dropout 0.05 \
		--use_expert_rules \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 12:00:00 \
		--run_name qwen3-0.6b-qlora-full-optimized

# Qwen3-0.6B - Hyperparameter Search (20 trials)
qwen3-0.6b-hp-search:
	python hyperparameter_search.py \
		--search_type random \
		--n_trials 20 \
		--base_output_dir /dss/mcmlscratch/0E/ra42qov2/models/qwen3-0.6b-hp-search \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 10:00:00 \
		--delay 1

# Qwen3-0.6B - Quick Hyperparameter Search (3 trials for testing)
qwen3-0.6b-hp-search-small:
	python hyperparameter_search.py \
		--search_type random \
		--n_trials 3 \
		--base_output_dir /dss/mcmlscratch/0E/ra42qov2/models/qwen3-0.6b-hp-search-small \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 08:00:00 \
		--delay 1


# Llama-3.3-70B - Full training (requires 2x 80GB GPUs with QLoRA, optimized hyperparameters)
llama-full:
	python end2end_finetuning.py \
		--model_id meta-llama/Llama-3.3-70B-Instruct \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/llama-3.3-70b-qlora-finetuned \
		--epochs 10 \
		--batch_size 1 \
		--gradient_accumulation_steps 8 \
		--learning_rate 2.5e-4 \
		--warmup_steps 10 \
		--weight_decay 0.1 \
		--max_grad_norm 0.5 \
		--label_smoothing 0.1 \
		--use_qlora \
		--lora_r 16 \
		--lora_alpha 64 \
		--lora_dropout 0.05 \
		--use_expert_rules \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 2 \
		--slurm_time 32:00:00 \
		--run_name llama-3.3-70b-qlora-full-optimized

# Llama-3.3-70B - Debug run with QLoRA
llama-debug:
	python end2end_finetuning.py \
		--model_id meta-llama/Llama-3.3-70B-Instruct \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/llama-3.3-70b-qlora-debug \
		--epochs 1 \
		--batch_size 1 \
		--gradient_accumulation_steps 8 \
		--learning_rate 1e-4 \
		--warmup_steps 5 \
		--use_qlora \
		--lora_r 8 \
		--lora_alpha 16 \
		--debug \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 03:00:00 \
		--run_name llama-3.3-70b-qlora-debug

# Qwen2.5-72B - Full training (requires 2x 80GB GPUs with QLoRA, optimized hyperparameters)
qwen2.5-72b-full:
	python end2end_finetuning.py \
		--model_id Qwen/Qwen2.5-72B-Instruct \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/qwen2.5-72b-qlora-finetuned \
		--epochs 10 \
		--batch_size 1 \
		--gradient_accumulation_steps 8 \
		--learning_rate 2.5e-4 \
		--warmup_steps 10 \
		--weight_decay 0.1 \
		--max_grad_norm 0.5 \
		--label_smoothing 0.1 \
		--use_qlora \
		--lora_r 16 \
		--lora_alpha 64 \
		--lora_dropout 0.05 \
		--use_expert_rules \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 2 \
		--slurm_time 32:00:00 \
		--run_name qwen2.5-72b-qlora-full-optimized

# Qwen2.5-72B - Debug run with QLoRA
qwen2.5-72b-debug:
	python end2end_finetuning.py \
		--model_id Qwen/Qwen2.5-72B-Instruct \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/qwen2.5-72b-qlora-debug \
		--epochs 1 \
		--batch_size 1 \
		--gradient_accumulation_steps 8 \
		--learning_rate 2e-4 \
		--warmup_steps 5 \
		--use_qlora \
		--lora_r 16 \
		--lora_alpha 64 \
		--lora_dropout 0.05 \
		--debug \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 2 \
		--slurm_time 03:00:00 \
		--run_name qwen2.5-72b-qlora-debug

# mT5-small (300M params, 512 token limit - may truncate long Latin texts)
mt5Small-debug:
	python end2end_finetuning.py \
		--model_id google/mt5-small \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/mt5-small-latin-debug \
		--epochs 1 \
		--batch_size 4 \
		--learning_rate 5e-5 \
		--warmup_steps 5 \
		--max_input_length 512 \
		--debug \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--run_name mt5-small-latin-debug

mt5Small-full:
	python end2end_finetuning.py \
		--model_id google/mt5-small \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/mt5-small-latin-finetuned \
		--epochs 30 \
		--batch_size 4 \
		--gradient_accumulation_steps 2 \
		--learning_rate 1e-4 \
		--warmup_steps 10 \
		--weight_decay 0.1 \
		--max_grad_norm 0.5 \
		--label_smoothing 0.1 \
		--max_input_length 512 \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 08:00:00 \
		--use_expert_rules \
		--run_name mt5-small-latin-full-optimized

# mT5-base (580M params, 512 token limit - better quality than small)
mt5Base-debug:
	python end2end_finetuning.py \
		--model_id google/mt5-base \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/mt5-base-latin-debug \
		--epochs 1 \
		--batch_size 4 \
		--learning_rate 3e-5 \
		--warmup_steps 5 \
		--max_input_length 512 \
		--debug \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 03:00:00 \
		--run_name mt5-base-latin-debug

mt5Base-full:
	python end2end_finetuning.py \
		--model_id google/mt5-base \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/mt5-base-latin-finetuned \
		--epochs 30 \
		--batch_size 2 \
		--gradient_accumulation_steps 4 \
		--learning_rate 1e-4 \
		--warmup_steps 10 \
		--weight_decay 0.1 \
		--max_grad_norm 0.5 \
		--label_smoothing 0.1 \
		--max_input_length 512 \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 12:00:00 \
		--use_expert_rules \
		--run_name mt5-base-latin-full-optimized

# LongT5 (supports up to 16K tokens - best for long Latin charters)
longT5-debug:
	python end2end_finetuning.py \
		--model_id google/long-t5-tglobal-base \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/long-t5-latin-debug \
		--epochs 1 \
		--batch_size 2 \
		--learning_rate 3e-5 \
		--warmup_steps 5 \
		--max_input_length 4096 \
		--debug \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 03:00:00 \
		--run_name long-t5-latin-debug

longT5-full:
	python end2end_finetuning.py \
		--model_id google/long-t5-tglobal-base \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/long-t5-latin-finetuned \
		--epochs 30 \
		--batch_size 2 \
		--gradient_accumulation_steps 4 \
		--learning_rate 1e-4 \
		--warmup_steps 10 \
		--weight_decay 0.1 \
		--max_grad_norm 0.5 \
		--label_smoothing 0.1 \
		--max_input_length 4096 \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 16:00:00 \
		--use_expert_rules \
		--run_name long-t5-latin-full-optimized

# MADLAD400-3B-MT (Google's multilingual T5, 3B params, 1024 token limit)
madlad-debug:
	python end2end_finetuning.py \
		--model_id google/madlad400-3b-mt \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/madlad400-3b-debug \
		--epochs 1 \
		--batch_size 1 \
		--gradient_accumulation_steps 4 \
		--learning_rate 3e-5 \
		--warmup_steps 5 \
		--max_input_length 1024 \
		--debug \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--run_name madlad400-3b-debug

madlad-full:
	python end2end_finetuning.py \
		--model_id google/madlad400-3b-mt \
		--dataset ../data/all.csv \
		--output_dir /dss/mcmlscratch/0E/ra42qov2/models/madlad400-3b-finetuned \
		--epochs 30 \
		--batch_size 1 \
		--gradient_accumulation_steps 8 \
		--learning_rate 1e-4 \
		--warmup_steps 10 \
		--weight_decay 0.1 \
		--max_grad_norm 0.5 \
		--label_smoothing 0.1 \
		--max_input_length 1024 \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 15:00:00 \
		--use_expert_rules \
		--run_name madlad400-3b-full-optimized

# Evaluate finetuned PhilTa model
eval-philta:
	python end2end_finetuning.py \
		--model_id ./models/philta-finetuned \
		--dataset ../data/all.csv \
		--eval_only \
		--use_expert_rules \
		--run_name philta-eval

# Clean up models and logs
clean:
	@echo "⚠️  This will delete all model checkpoints and logs!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf ./models/*; \
		rm -rf ./logs/*; \
		echo "✓ Cleaned models/ and logs/"; \
	fi
