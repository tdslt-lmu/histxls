# Cross-lingual Text Summarization of Historical Documents

## Abstract

The Bavarian Academy of Sciences and Humanities (BAdW) is producing a critical edition of approximately 2,600 medieval Latin charters issued by Emperor Frederick II (1194–1250). A key step in this process is writing *Kopfregesten*—standardized single-sentence German summaries that follow strict diplomatic conventions regarding structure, content, and naming. Creating these summaries manually requires senior Latin scholars and remains a major bottleneck in the multi-decade edition project.

This thesis investigates whether large language models (LLMs) can automate this crosslingual summarization task. Three paradigms are compared on a corpus of 1,700 charter–Kopfregest pairs: few-shot prompting with large instruction-following models, modular pipelines (translate-then-summarize and summarize-then-translate), and end-to-end finetuning with parameter-efficient methods. In total, 17 model–approach combinations are evaluated across encoder–decoder and decoder-only architectures ranging from 0.3B to 72B parameters.

To assess output quality beyond standard NLP metrics, a specialized evaluation framework is developed that organizes 14 metrics into five equally weighted categories—Lexical, Semantic, Distributional, Constraints, and Efficiency—and aggregates them into a single composite score in $[0, 1]$, with higher values indicating better overall output quality. The Constraints category uses an LLM-as-a-Judge approach to check adherence to six diplomatic rules, capturing violations that surface-level metrics miss.

End-to-end finetuning proves most effective: a finetuned Qwen2.5-72B achieves the highest overall score (0.744), while a finetuned 8B-parameter model already surpasses the best 70B few-shot result. Few-shot prompting provides a competitive training-free baseline for large models (≥ 0.70) but fails for small ones. Pipeline approaches perform substantially worse (≤ 0.511) due to error propagation and loss of context between stages. Importantly, finetuned models in the 3–8B range deliver strong results on consumer hardware, making them directly deployable for the BAdW's resource-constrained workflow as draft-generation tools with human expert review.

## Repository Structure

```
.
├── thesis/                          # Thesis document (tex/pdf)
├── data/                            # Pointer to the actual data
├── results/                         # Pointer to the actual results
├── code/
│   ├── DataExtraction/              # Chapter 3 – Dataset extraction from source documents
│   │   ├── RegexExtraction/         #   Rule-based extraction via regex from DOCX files
│   │   ├── LLMExtraction/           #   AI-based extraction via Gemini API from HTML files
│   │   ├── csv/                     #   Extracted CSV outputs (regex & LLM)
│   │   ├── data/                    #   Source documents (DOCX, HTML)
│   │   └── helper/                  #   Utility scripts
│   └── ApproachEvaluation/          # Chapter 4 – Methodology
│       ├── 01_FewShotPrompting/     #   Few-shot prompting with large LLMs
│       │   ├── experiment_results/  #     Metric results per experiment
│       │   └── predictions/         #     Model predictions (CSV)
│       ├── 02_PipelineApproach/     #   Two-stage pipelines (STT & TTS)
│       │   ├── experiment_results/
│       │   └── predictions/
│       ├── 03_Finetuning/           #   End-to-end finetuning (PEFT)
│       │   ├── experiment_results/
│       │   └── predictions/
│       ├── global_experiment_results/  # Aggregated cross-approach results
│       ├── data/                    #   Dataset (all.csv)
│       └── src/                     #   Shared modules
│           ├── evaluator.py         #     Evaluation framework
│           ├── model_wrappers.py    #     Unified model loading & inference
│           └── slurm_manager.py     #     SLURM job submission & management
└── README.md
```

## Approaches

### 1. Few-Shot Prompting
Prompt-based approach using large LLMs with few-shot examples to directly generate German Kopfregesten from Latin charter texts.
- Script: [`code/ApproachEvaluation/01_FewShotPrompting/`](code/ApproachEvaluation/01_FewShotPrompting/)

### 2. Pipeline Approach
Two-stage decomposition of the task:
- **STT** (Summarize-Then-Translate): Latin → Latin summary → German summary
- **TTS** (Translate-Then-Summarize): Latin → German translation → German summary
- Scripts: [`code/ApproachEvaluation/02_PipelineApproach/`](code/ApproachEvaluation/02_PipelineApproach/)

### 3. End-to-End Finetuning
Parameter-efficient finetuning of encoder–decoder and decoder-only models on the Latin-to-German summarization task.
- Scripts: [`code/ApproachEvaluation/03_Finetuning/`](code/ApproachEvaluation/03_Finetuning/)

## Results

The global results across all approaches can be found in:
`code/ApproachEvaluation/global_experiment_results`

Per-approach results and predictions are stored in the respective `experiment_results/` and `predictions/` subdirectories. See [`results/README.md`](results/README.md) for a full overview.

## Getting Started

Both code subfolders contain dedicated READMEs with important information on setup and usage:

- [`code/DataExtraction/README.md`](code/DataExtraction/README.md) — Setup and instructions for the data extraction pipeline
- [`code/ApproachEvaluation/README.md`](code/ApproachEvaluation/README.md) — Setup and instructions for the approach evaluation experiments
