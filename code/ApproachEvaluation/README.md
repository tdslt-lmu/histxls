# Master Thesis – Crosslingual Summarization of Medieval Latin Charters

This folder is about the following Chapter of my Thesis: 4 - Methodology.

The thesis explores three main approaches to tackle this crosslingual summarization task, comparing their effectiveness using a self-developed evaluation framework.

## Approaches

### 01 – Few-Shot Prompting
Prompt-based approach using large language models (LLMs) with few-shot examples to directly generate German Kopfregesten from Latin texts.  
Scripts: [`01_FewShotPrompting/`](01_FewShotPrompting/)

### 02 – Pipeline Approach
Two-stage pipeline approaches that decompose the task:
- **STT (Summarize-Then-Translate)**: Summarize Latin → Latin, then translate Latin → German
- **TTS (Translate-Then-Summarize)**: Translate Latin → German, then summarize German → German

Scripts: [`02_PipelineApproach/`](02_PipelineApproach/)

### 03 – Fine-Tuning
End-to-end fine-tuning of sequence-to-sequence models (mT5, MADLAD, Qwen) on the Latin-to-German summarization task.  
Scripts: [`03_Finetuning/`](03_Finetuning/)

## Project Structure

| Directory | Contents |
|---|---|
| `01_FewShotPrompting/` | Few-shot prompting experiments with various LLMs |
| `02_PipelineApproach/` | Two-stage pipeline implementations (STT/TTS) |
| `03_Finetuning/` | Fine-tuning scripts and hyperparameter optimization |
| `data/` | Dataset file with all the charter texts and reference Kopfregesten |
| `src/` | Shared modules (evaluator, model wrappers, SLURM manager) |
| `results/` | Aggregated overall result csv's |

## Setup

Python 3.10+, with CUDA-enabled GPU:

```bash
# Create conda environment
conda create -n llm-thesis python=3.10
conda activate llm-thesis

# Install dependencies
pip install -r requirements.txt

# Login to Weights & Biases for experiment tracking
wandb login
```

## Usage

> **Note:** This code is intended to be executed on the LRZ login node, as it will spawn training jobs in the LRZ cluster via SLURM.

Each approach folder contains its own `Makefile` with predefined targets for running specific experiments. Example:

```bash
cd 01_FewShotPrompting
make qwen3-0.6b  # This will initiate the FewShotPrompting Experiment for the qwen3 model. In the makefile itself you can control the hyperparameter
```