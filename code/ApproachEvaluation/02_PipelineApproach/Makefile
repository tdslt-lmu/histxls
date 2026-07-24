.PHONY: help tts-nllb tts-madlad tts-export-full stt-latin stt-longt5 stt-mt5 all clean

# Default target
help:
	@echo "Available targets:"
	@echo ""
	@echo "TTS Pipeline (Translate-Then-Summarize):"
	@echo "  tts-nllb       - TTS with NLLB-200 translator + mT5-German summarizer (40GB GPU, 2h)"
	@echo "  tts-madlad     - TTS with MADLAD-400 translator + mT5-German summarizer (40GB GPU, 2h)"
	@echo ""
	@echo "STT Pipeline (Summarize-Then-Translate):"
	@echo "  stt-latin      - STT with LatinSummarizerModel + NLLB translator (40GB GPU, 2h)"
	@echo "  stt-longt5     - STT with LongT5-Large + NLLB translator (40GB GPU, 2h)"
	@echo "  stt-mt5        - STT with mT5-XLSum + NLLB translator (40GB GPU, 2h)"
	@echo ""
	@echo "Utility:"
	@echo "  all            - Run all TTS and STT pipelines"
	@echo "  clean          - Remove log files"

# ================================================================
# TTS Pipeline: Translate-Then-Summarize
# ================================================================

# TTS with NLLB-200-3.3B (translator) → mT5-German (summarizer)
tts-nllb:
	python tts_pipeline.py \
		--translator facebook/nllb-200-3.3B \
		--summarizer ml6team/mt5-small-german-finetune-mlsum \
		--dataset ../data/all.csv \
		--n_samples 150 \
		--max_length_translation 1024 \
		--max_length_summary 256 \
		--batch_size 8 \
		--prompt_version "3.0" \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--slurm_job_name tts-nllb-mt5 \
		--slurm_log_dir logs \
		--run_name tts-nllb200-mt5de

# TTS with MADLAD-400-3B (translator) → mT5-German (summarizer)
tts-madlad:
	python tts_pipeline.py \
		--translator google/madlad400-3b-mt \
		--summarizer ml6team/mt5-small-german-finetune-mlsum \
		--dataset ../data/all.csv \
		--n_samples 150 \
		--max_length_translation 1024 \
		--max_length_summary 256 \
		--batch_size 8 \
		--prompt_version "3.0" \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--slurm_job_name tts-madlad-mt5 \
		--slurm_log_dir logs \
		--run_name tts-madlad400-mt5de

# TTS Full Dataset Export (for finetuning data generation)
tts-export-full:
	python tts_pipeline.py \
		--translator google/madlad400-3b-mt \
		--summarizer ml6team/mt5-small-german-finetune-mlsum \
		--dataset ../data/all.csv \
		--use_full_dataset \
		--export_csv \
		--csv_output ../data/tts_finetuning_data_full.csv \
		--batch_size 8 \
		--prompt_version "3.0" \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 06:00:00 \
		--slurm_job_name tts-full-export \
		--slurm_log_dir logs \
		--run_name tts-full-dataset-export

# ================================================================
# STT Pipeline: Summarize-Then-Translate
# ================================================================

# STT with LatinSummarizerModel (summarizer) → NLLB-200 (translator)
stt-latin:
	python stt_pipeline.py \
		--summarizer LatinNLP/LatinSummarizerModel \
		--translator facebook/nllb-200-3.3B \
		--dataset ../data/all.csv \
		--n_samples 150 \
		--max_length_summary 256 \
		--max_length_translation 256 \
		--batch_size 8 \
		--prompt_version "3.0" \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--slurm_job_name stt-latin-nllb \
		--slurm_log_dir logs \
		--run_name stt-latin-nllb

# STT with LongT5-Large (summarizer) → NLLB-200 (translator)
stt-longt5:
	python stt_pipeline.py \
		--summarizer google/long-t5-tglobal-large \
		--translator facebook/nllb-200-3.3B \
		--dataset ../data/all.csv \
		--n_samples 150 \
		--max_length_summary 256 \
		--max_length_translation 256 \
		--batch_size 8 \
		--prompt_version "3.0" \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--slurm_job_name stt-longt5-nllb \
		--slurm_log_dir logs \
		--run_name stt-longt5-nllb

# STT with mT5-XLSum (summarizer) → NLLB-200 (translator)
stt-mt5:
	python stt_pipeline.py \
		--summarizer csebuetnlp/mT5_multilingual_XLSum \
		--translator facebook/nllb-200-3.3B \
		--dataset ../data/all.csv \
		--n_samples 150 \
		--max_length_summary 256 \
		--max_length_translation 256 \
		--batch_size 8 \
		--prompt_version "3.0" \
		--run_mode slurm \
		--slurm_partition mcml-dgx-a100-40x8 \
		--slurm_gpus 1 \
		--slurm_time 02:00:00 \
		--slurm_job_name stt-mt5-nllb \
		--slurm_log_dir logs \
		--run_name stt-mt5xlsum-nllb

# ================================================================
# Utility targets
# ================================================================

# Run all TTS and STT pipelines sequentially
all: tts-nllb tts-madlad stt-latin stt-longt5 stt-mt5

# Clean log files
clean:
	@echo "Cleaning logs..."
	rm -f logs/*.out logs/*.err
	@echo "Done!"
