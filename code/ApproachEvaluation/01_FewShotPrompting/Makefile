.PHONY: help qwen3-0.6b qwen3-8b qwen2.5-72b llama-70b all clean

# Default target
help:
	@echo "Available targets:"
	@echo "  qwen3-0.6b    - Run few-shot prompting with Qwen3-0.6B (40GB GPU)"
	@echo "  qwen3-8b      - Run few-shot prompting with Qwen3-8B (80GB GPU)"
	@echo "  qwen2.5-72b   - Run few-shot prompting with Qwen2.5-72B-Instruct (80GB GPU, 2 GPUs)"
	@echo "  llama-70b     - Run few-shot prompting with Llama-3.3-70B-Instruct (80GB GPU, 2 GPUs)"
	@echo "  all           - Run all models sequentially"
	@echo "  clean         - Remove log files"

# Qwen3-0.6B - Small model (40GB GPU)
qwen3-0.6b:
	python fewshot_summarization.py \
		--model_id "Qwen/Qwen3-0.6B" \
		--model_type causal \
		--n_shots 10 \
		--n_test_samples 150 \
		--max_new_tokens 256 \
		--max_context_length 32768 \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 08:00:00 \
		--system_prompt_version "4.0"

# Qwen3-8B - Medium model (80GB GPU)
qwen3-8b:
	python fewshot_summarization.py \
		--model_id "Qwen/Qwen3-8B" \
		--model_type causal \
		--n_shots 10 \
		--n_test_samples 150 \
		--max_new_tokens 256 \
		--max_context_length 32768 \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 1 \
		--slurm_time 08:00:00 \
		--system_prompt_version "1.0"

# Qwen2.5-72B-Instruct - Large model (80GB GPU, 2 GPUs)
qwen2.5-72b:
	python fewshot_summarization.py \
		--model_id "Qwen/Qwen2.5-72B-Instruct" \
		--model_type causal \
		--n_shots 10 \
		--n_test_samples 150 \
		--max_new_tokens 256 \
		--max_context_length 32768 \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 2 \
		--slurm_time 10:00:00 \
		--system_prompt_version "4.0"

# Llama-3.3-70B-Instruct - Large model (80GB GPU, 2 GPUs)
llama-70b:
	python fewshot_summarization.py \
		--model_id "meta-llama/Llama-3.3-70B-Instruct" \
		--model_type causal \
		--n_shots 10 \
		--n_test_samples 150 \
		--max_new_tokens 256 \
		--max_context_length 32768 \
		--run_mode slurm \
		--slurm_partition mcml-hgx-a100-80x4 \
		--slurm_gpus 2 \
		--slurm_time 10:00:00 \
		--system_prompt_version "4.0"

# Run all models
all: qwen3-0.6b qwen3-8b qwen2.5-72b llama-70b

# Clean log files
clean:
	@echo "Cleaning logs..."
	rm -f logs/*.out logs/*.err
	@echo "Done!"
