#!/usr/bin/env python3
"""
TTS Pipeline for Crosslingual Summarization (Translate-Then-Summarize)
Uses specific models for Translation and Summarization steps.

Step 1: Translate Latin -> German (using MADLAD-400 or NLLB)
Step 2: Summarize German -> German Summary (using a German summarization model)
"""

import os
import sys
import argparse
import torch
import gc
import wandb
import json
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluator import SummarizationEvaluator, EvaluationMetrics
from src.slurm_manager import SlurmConfig, SlurmJobSubmitter

@dataclass
class TTSConfig:
    translator_model_id: str
    summarizer_model_id: str
    dataset_path: str
    n_test_samples: int = 50
    max_length_translation: int = 1024
    max_length_summary: int = 256
    batch_size: int = 4
    wandb_project: str = "crosslingual-summarization-thesis"
    wandb_run_name: str = "tts-pipeline"
    # Evaluation configuration (matching fewshot_summarization.py)
    bertscore_model: str = "magistermilitum/bert_medieval_multilingual"
    mauve_model: str = "magistermilitum/bert_medieval_multilingual"
    # Prompt/Filter options
    prompts_file: str = "pipeline_prompts.json"
    prompt_version: Optional[str] = None
    # CSV Export options
    export_csv: bool = False
    csv_output_path: Optional[str] = None
    use_full_dataset: bool = False  # If True, process entire dataset instead of just test split
    # SLURM options
    run_mode: str = "local"
    slurm_partition: str = "mcml-dgx-a100-40x8"
    slurm_gpus: int = 1
    slurm_time: str = "02:00:00"
    slurm_job_name: Optional[str] = None
    slurm_log_dir: str = "logs"
    slurm_conda_env: str = "llm-thesis"
    slurm_email: Optional[str] = None

class TTSPipeline:
    def __init__(self, config: TTSConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = []
        self.whitelisted_charter_ids = []
        self.blacklisted_charter_ids = []
        
        print(f"Running on device: {self.device}")

    def load_data(self):
        """Loads and prepares the dataset"""
        print(f"Loading dataset: {self.config.dataset_path}")
        dataset = load_dataset("csv", data_files=self.config.dataset_path, engine="python")['train']
        
        if self.config.use_full_dataset:
            # Use entire dataset
            print("Using ENTIRE dataset (no Train/Test Split)")
            working_dataset = dataset
        else:
            # 90/10 Split (for testing/evaluation)
            train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
            working_dataset = train_test_split['test']
            print("Using test split (10% of dataset)")
        
        # Filter whitelist if provided
        if self.whitelisted_charter_ids:
            print(f"Whitelist active! Filtering for {len(self.whitelisted_charter_ids)} Charter IDs...")
            working_dataset = working_dataset.filter(
                lambda x: str(x['nummer']) in self.whitelisted_charter_ids
            )
        elif self.blacklisted_charter_ids:
            print(f"Blacklist active! Removing {len(self.blacklisted_charter_ids)} Charter IDs...")
            working_dataset = working_dataset.filter(
                lambda x: str(x['nummer']) not in self.blacklisted_charter_ids
            )
        
        # Select subset (only if not using full dataset)
        if not self.config.use_full_dataset:
            n_test = min(self.config.n_test_samples, len(working_dataset))
            working_dataset = working_dataset.shuffle(seed=42).select(range(n_test))
        
        print(f"Dataset prepared: {len(working_dataset)} samples")
        return working_dataset

    def run_translation(self, texts: List[str]) -> List[str]:
        """Step 1: Translate Latin texts to German"""
        print(f"\n=== Step 1: Translation (Latin -> German) ===")
        print(f"Model: {self.config.translator_model_id}")
        
        # Use float16 for large models to save memory
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.translator_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.translator_model_id,
            torch_dtype=torch_dtype
        ).to(self.device)
        model.eval()
        
        translations = []
        
        # Check model type for special handling
        is_nllb = "nllb" in self.config.translator_model_id.lower()
        is_madlad = "madlad" in self.config.translator_model_id.lower()
        
        if is_nllb:
            # NLLB Language Codes
            src_lang = "lat_Latn"  # Latin
            tgt_lang = "deu_Latn"  # German
            
            if hasattr(tokenizer, "src_lang"):
                tokenizer.src_lang = src_lang
                
            print(f"NLLB Mode - Source Lang: {src_lang}, Target Lang: {tgt_lang}")
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
            print(f"Forced BOS Token ID for {tgt_lang}: {forced_bos_token_id}")
            text_prefix = ""
        elif is_madlad:
            # MADLAD-400 uses a special prefix format: <2de> for German target
            print(f"MADLAD Mode - Using '<2de>' prefix for German target")
            forced_bos_token_id = None
            text_prefix = "<2de> "  # Target language prefix
        else:
            # For other models, no special language codes needed
            print(f"Standard translation mode (no language codes needed)")
            forced_bos_token_id = None
            text_prefix = ""
        
        for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Translating"):
            batch_texts = texts[i:i+self.config.batch_size]
            
            # Add prefix for MADLAD
            if text_prefix:
                batch_texts = [text_prefix + (text if text is not None else "") for text in batch_texts]
            
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                generate_kwargs = {
                    "max_new_tokens": self.config.max_length_translation,
                    "num_beams": 4
                }
                if forced_bos_token_id is not None:
                    generate_kwargs["forced_bos_token_id"] = forced_bos_token_id
                    
                outputs = model.generate(**inputs, **generate_kwargs)
            
            batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(batch_translations)
            
        # Cleanup
        print("Cleaning up Translator model...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return translations

    def run_summarization(self, texts: List[str]) -> List[str]:
        """Step 2: Summarize German texts to German summaries"""
        print(f"\n=== Step 2: Summarization (German -> German Summary) ===")
        print(f"Model: {self.config.summarizer_model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.summarizer_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.summarizer_model_id).to(self.device)
        model.eval()
        
        # Determine task prefix based on model type
        model_id_lower = self.config.summarizer_model_id.lower()
        if "t5" in model_id_lower:
            # mT5 and T5 models need a task prefix
            if "mt5" in model_id_lower:
                task_prefix = "summarize: "
            else:
                task_prefix = "summarize: "
            print(f"  Using task prefix: '{task_prefix}'")
        elif "bart" in model_id_lower:
            task_prefix = ""  # BART doesn't need a prefix
        else:
            task_prefix = ""
        
        summaries = []
        
        for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Summarizing"):
            batch_texts = texts[i:i+self.config.batch_size]
            
            # Add task prefix if needed (with None-safety)
            if task_prefix:
                batch_texts = [task_prefix + (text if text is not None else "") for text in batch_texts]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=self.config.max_length_summary,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode
            batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries.extend(batch_summaries)
        
        # Cleanup to free VRAM
        print("Cleaning up Summarizer model...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return summaries

    def export_to_csv(self):
        """Exports results to CSV file for future finetuning"""
        print("\n=== Exporting to CSV ===")
        
        # Generate default filename if not provided
        if self.config.csv_output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            translator_short = self.config.translator_model_id.split("/")[-1]
            summarizer_short = self.config.summarizer_model_id.split("/")[-1]
            filename = f"tts_results_{translator_short}_{summarizer_short}_{timestamp}.csv"
            output_path = Path(__file__).parent / filename
        else:
            output_path = Path(self.config.csv_output_path)
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Reorder columns for clarity
        column_order = ["nummer", "latin_text", "german_translation", "reference", "prediction"]
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding="utf-8")
        
        print(f"✓ CSV exported: {output_path}")
        print(f"  Total samples: {len(df)}")
        
        return output_path

    def run(self):
        # Initialize W&B (skip if only exporting full dataset)
        if not self.config.use_full_dataset:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
        
        # Load Data
        test_data = self.load_data()
        
        # Filter out None/empty values
        print("\n=== Data Cleaning ===")
        original_size = len(test_data)
        test_data = test_data.filter(lambda x: x['latin_text'] is not None and x['latin_text'].strip() != "")
        filtered_size = len(test_data)
        if filtered_size < original_size:
            print(f"Removed {original_size - filtered_size} samples with empty/None latin_text")
        
        input_texts = test_data['latin_text']
        references = test_data['summary_de']
        
        # Step 1: Translate Latin -> German
        german_translations = self.run_translation(input_texts)
        
        # Step 2: Summarize German -> German Summary
        german_summaries = self.run_summarization(german_translations)
        
        # Store results with charter IDs
        charter_ids = test_data['nummer']  # Charter IDs from dataset
        for i in range(len(input_texts)):
            self.results.append({
                "nummer": charter_ids[i],
                "latin_text": input_texts[i],
                "german_translation": german_translations[i],
                "prediction": german_summaries[i],
                "reference": references[i]
            })
        
        if self.config.use_full_dataset:
            # Skip evaluation for full dataset - just export CSV
            print("\n=== Skipping Evaluation (full dataset mode) ===")
            print("Directly exporting to CSV...")
            
            if self.config.export_csv:
                self.export_to_csv()
            else:
                print("Warning: use_full_dataset is True but export_csv is False. No output will be saved!")
            
            print("Dataset processing finished successfully!")
        else:
            # Evaluate (only for test samples)
            print("\n=== Evaluation ===")
            evaluator = SummarizationEvaluator(
                bertscore_model=self.config.bertscore_model,
                bertscore_num_layers=12,
                mauve_model=self.config.mauve_model,
                verbose=True
            )
            
            # Get tokenizer for tokenizer metrics (use the summarizer's tokenizer)
            tokenizer_for_eval = self.summarizer_tokenizer
            
            metrics = evaluator.evaluate(
                predictions=german_summaries,
                references=references,
                inputs=input_texts,
                use_expert_rules=True,
                tokenizer_id_or_obj=tokenizer_for_eval
            )
            
            # Log to W&B
            wandb.log(metrics.to_dict())
            
            # Log Table with intermediate step
            table = wandb.Table(columns=["Charter ID", "Input (Latin)", "German Translation (Step 1)", "German Summary (Step 2)", "Reference"])
            for res in self.results:
                table.add_data(
                    res["nummer"],
                    res["latin_text"][:500],
                    res["german_translation"][:500],  # Truncate for display
                    res["prediction"],
                    res["reference"]
                )
            wandb.log({"predictions": table})
            
            wandb.finish()
            
            # Export to CSV if requested
            if self.config.export_csv:
                self.export_to_csv()
            
            print("Experiment finished successfully!")

def build_slurm_python_args(config: TTSConfig) -> List[str]:
    """Builds Python arguments for SLURM submission"""
    args = [
        "--translator", config.translator_model_id,
        "--summarizer", config.summarizer_model_id,
        "--dataset", config.dataset_path,
        "--n_samples", str(config.n_test_samples),
        "--batch_size", str(config.batch_size),
        "--run_name", config.wandb_run_name,
        "--prompts_file", config.prompts_file,
        "--run_mode", "local"  # Important: prevent recursive submission
    ]
    
    if config.prompt_version:
        args.extend(["--prompt_version", config.prompt_version])
    
    if config.export_csv:
        args.append("--export_csv")
    
    if config.csv_output_path:
        args.extend(["--csv_output", config.csv_output_path])
    
    if config.use_full_dataset:
        args.append("--use_full_dataset")
        
    return args

def main():
    parser = argparse.ArgumentParser(description="TTS Pipeline (Translate-Then-Summarize)")
    parser.add_argument("--translator", type=str, default="google/madlad400-3b-mt",
                        help="Translation model (Latin -> German). Options: google/madlad400-3b-mt, facebook/nllb-200-3.3B")
    parser.add_argument("--summarizer", type=str, default="ml6team/mt5-small-german-finetune-mlsum",
                        help="German summarization model")
    parser.add_argument("--dataset", type=str, default="/dss/dsshome1/0E/ra42qov2/development/data/all.csv")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--run_name", type=str, default="tts-nllb-mt5de")
    
    # CSV Export args
    parser.add_argument("--export_csv", action="store_true", help="Export results to CSV")
    parser.add_argument("--csv_output", type=str, default=None, help="CSV output path (auto-generated if not specified)")
    parser.add_argument("--use_full_dataset", action="store_true", help="Process entire dataset instead of just test split")
    
    # Prompt args
    parser.add_argument("--prompts_file", type=str, default="pipeline_prompts.json")
    parser.add_argument("--prompt_version", type=str, default=None)
    
    # SLURM args
    parser.add_argument("--run_mode", type=str, choices=["local", "slurm"], default="local")
    parser.add_argument("--slurm_partition", type=str, default="mcml-dgx-a100-40x8")
    parser.add_argument("--slurm_gpus", type=int, default=1)
    parser.add_argument("--slurm_time", type=str, default="02:00:00")
    parser.add_argument("--slurm_job_name", type=str, default=None)
    parser.add_argument("--slurm_log_dir", type=str, default="logs")
    parser.add_argument("--slurm_conda_env", type=str, default="llm-thesis")
    parser.add_argument("--slurm_email", type=str, default=None)
    
    args = parser.parse_args()
    
    config = TTSConfig(
        translator_model_id=args.translator,
        summarizer_model_id=args.summarizer,
        dataset_path=args.dataset,
        n_test_samples=args.n_samples,
        batch_size=args.batch_size,
        wandb_run_name=args.run_name,
        prompts_file=args.prompts_file,
        prompt_version=args.prompt_version,
        export_csv=args.export_csv,
        csv_output_path=args.csv_output,
        use_full_dataset=args.use_full_dataset,
        run_mode=args.run_mode,
        slurm_partition=args.slurm_partition,
        slurm_gpus=args.slurm_gpus,
        slurm_time=args.slurm_time,
        slurm_job_name=args.slurm_job_name,
        slurm_log_dir=args.slurm_log_dir,
        slurm_conda_env=args.slurm_conda_env,
        slurm_email=args.slurm_email
    )
    
    if args.run_mode == "slurm":
        # Submit to SLURM
        job_name = args.slurm_job_name or f"tts-{args.run_name}"
        
        slurm_config = SlurmConfig(
            partition=args.slurm_partition,
            num_gpus=args.slurm_gpus,
            time_limit=args.slurm_time,
            job_name=job_name,
            log_dir=args.slurm_log_dir,
            conda_env=args.slurm_conda_env,
            email=args.slurm_email
        )
        
        python_args = build_slurm_python_args(config)
        
        print("=== SLURM Job Submission ===")
        print(f"Translator: {config.translator_model_id}")
        print(f"Summarizer: {config.summarizer_model_id}")
        print(f"Partition: {slurm_config.partition}")
        print()
        
        submitter = SlurmJobSubmitter(
            script_path=str(Path(__file__).resolve()),
            python_args=python_args,
            slurm_config=slurm_config
        )
        job_id = submitter.submit()
        
    else:
        # Run locally
        pipeline = TTSPipeline(config)
        pipeline.run()

if __name__ == "__main__":
    main()
