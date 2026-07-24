#!/usr/bin/env python3
"""
STT Pipeline for Crosslingual Summarization (Summarize-Then-Translate)
Uses specific models for Summarization and Translation steps.

Step 1: Summarize Latin -> Latin (using LatinNLP/LatinSummarizerModel)
Step 2: Translate Latin Summary -> German Kopfregest
"""

import os
import sys
import argparse
import torch
import gc
import wandb
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration, MT5Tokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluator import SummarizationEvaluator, EvaluationMetrics
from src.slurm_manager import SlurmConfig, SlurmJobSubmitter

@dataclass
class STTConfig:
    summarizer_model_id: str
    translator_model_id: str
    dataset_path: str
    n_test_samples: int = 50
    max_length_summary: int = 256
    max_length_translation: int = 256
    batch_size: int = 4
    wandb_project: str = "crosslingual-summarization-thesis"
    wandb_run_name: str = "stt-pipeline"
    # Evaluation configuration
    bertscore_model: str = "magistermilitum/bert_medieval_multilingual"
    mauve_model: str = "magistermilitum/bert_medieval_multilingual"
    # Prompt/Filter options
    prompts_file: str = "pipeline_prompts.json"
    prompt_version: Optional[str] = None
    # SLURM options
    run_mode: str = "local"
    slurm_partition: str = "mcml-dgx-a100-40x8"
    slurm_gpus: int = 1
    slurm_time: str = "02:00:00"
    slurm_job_name: Optional[str] = None
    slurm_log_dir: str = "logs"
    slurm_conda_env: str = "llm-thesis"
    slurm_email: Optional[str] = None

class STTPipeline:
    def __init__(self, config: STTConfig):
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
        
        # 90/10 Split
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        test_dataset = train_test_split['test']
        
        # Filter whitelist if provided
        if self.whitelisted_charter_ids:
            print(f"Whitelist active... Filtering for {len(self.whitelisted_charter_ids)} Charter IDs...")
            test_dataset = test_dataset.filter(
                lambda x: str(x['nummer']) in self.whitelisted_charter_ids
            )
        elif self.blacklisted_charter_ids:
            print(f"Blacklist active... Removing {len(self.blacklisted_charter_ids)} Charter IDs...")
            test_dataset = test_dataset.filter(
                lambda x: str(x['nummer']) not in self.blacklisted_charter_ids
            )
        
        # Select subset
        n_test = min(self.config.n_test_samples, len(test_dataset))
        test_dataset_subset = test_dataset.shuffle(seed=42).select(range(n_test))
        
        print(f"Dataset prepared: {len(test_dataset_subset)} samples")
        return test_dataset_subset

    def run_summarization(self, texts: List[str]) -> List[str]:
        """Step 1: Summarize Latin texts to Latin summaries"""
        print(f"\n=== Step 1: Summarization (Latin -> Latin) ===")
        print(f"Model: {self.config.summarizer_model_id}")
        
        # Special handling for LatinNLP/LatinSummarizerModel which is an mT5 model
        # The repo structure is non-standard: tokenizer in a subfolder, weights as .pt file
        if "LatinSummarizerModel" in self.config.summarizer_model_id:
            print("  Detected LatinSummarizerModel, using custom loading...")
            from huggingface_hub import hf_hub_download
            from transformers import T5Tokenizer
            
            # Tokenizer is in the subfolder
            tokenizer_subfolder = "final_pipeline/no_stanza/mt5-small-en-la-translation-final_no_stanza-last_epoch"
            print(f"  Loading tokenizer from subfolder: {tokenizer_subfolder}")
            tokenizer = T5Tokenizer.from_pretrained(
                self.config.summarizer_model_id, 
                subfolder=tokenizer_subfolder
            )
            
            # Model weights are stored as a .pt file
            pt_file_path = "final_pipeline/no_stanza/mt5-small-en-la-translation-final_no_stanza-last_epoch.pt"
            print(f"  Downloading weights from: {pt_file_path}")
            weights_path = hf_hub_download(
                repo_id=self.config.summarizer_model_id,
                filename=pt_file_path
            )
            
            # Load the state dict
            state_dict = torch.load(weights_path, map_location="cpu")
            
            # Check the structure of keys to determine how to load
            sample_keys = list(state_dict.keys())[:5]
            print(f"  Sample keys from checkpoint: {sample_keys}")
            
            # The checkpoint has 'base_model.model.' prefix - we need to strip it
            # to load into a vanilla MT5 model
            if any(k.startswith("base_model.model.") for k in state_dict.keys()):
                print("  Detected 'base_model.model.' prefix, stripping for vanilla MT5 loading...")
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Skip LoRA-specific keys (they contain 'lora_A', 'lora_B')
                    if "lora_A" in key or "lora_B" in key:
                        continue
                    # Strip the prefix
                    if key.startswith("base_model.model."):
                        new_key = key.replace("base_model.model.", "")
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            # Get vocab size from checkpoint
            if "shared.weight" in state_dict:
                checkpoint_vocab_size = state_dict["shared.weight"].shape[0]
            else:
                checkpoint_vocab_size = None
            
            # Initialize base model
            print("  Initializing MT5-small base model...")
            model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
            
            # Resize embeddings (if needed)
            if checkpoint_vocab_size and checkpoint_vocab_size != model.config.vocab_size:
                print(f"  Resizing model embeddings from {model.config.vocab_size} to {checkpoint_vocab_size}...")
                model.resize_token_embeddings(checkpoint_vocab_size)
            
            # Load the state dict
            load_result = model.load_state_dict(state_dict, strict=False)
            print(f"  Missing keys: {len(load_result.missing_keys)}")
            print(f"  Unexpected keys: {len(load_result.unexpected_keys)}")
            if load_result.missing_keys:
                print(f"  Sample missing: {load_result.missing_keys[:5]}")
            if load_result.unexpected_keys:
                print(f"  Sample unexpected: {load_result.unexpected_keys[:5]}")
            
            model = model.to(self.device)
            print("  LatinSummarizerModel loaded successfully")
            # T5/mT5 models typically need a task prefix
            task_prefix = "summarize: "
        else:
            # Standard loading for other models
            tokenizer = AutoTokenizer.from_pretrained(self.config.summarizer_model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.config.summarizer_model_id).to(self.device)
            
            # T5-based models need a task prefix
            if "t5" in self.config.summarizer_model_id.lower():
                task_prefix = "summarize: "
                print(f"  Using task prefix for T5 model: '{task_prefix}'")
            else:
                task_prefix = ""
            
        model.eval()
        
        summaries = []
        
        for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Summarizing"):
            batch_texts = texts[i:i+self.config.batch_size]
            
            # Add task prefix if needed (for T5/mT5 models)
            if task_prefix:
                batch_texts = [task_prefix + text for text in batch_texts]
            
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

    def run_translation(self, texts: List[str]) -> List[str]:
        """Step 2: Translate Latin summaries to German"""
        print(f"\n=== Step 2: Translation (Latin -> German) ===")
        print(f"Model: {self.config.translator_model_id}")
        
        # Use float16 for NLLB to save memory and speed up inference (like in the example)
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.translator_model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.translator_model_id,
            torch_dtype=torch_dtype
        ).to(self.device)
        model.eval()
        
        translations = []
        
        # NLLB Language Codes (Required for NLLB)
        src_lang = "lat_Latn"  # Latin
        tgt_lang = "deu_Latn"  # German
        
        # Ensure tokenizer has correct src_lang
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_lang
            
        print(f"Source Lang: {src_lang}, Target Lang: {tgt_lang}")
        
        # Get forced_bos_token_id for target language
        # NllbTokenizerFast uses convert_tokens_to_ids, not lang_code_to_id
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        print(f"Forced BOS Token ID for {tgt_lang}: {forced_bos_token_id}")
        
        for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Translating"):
            batch_texts = texts[i:i+self.config.batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=self.config.max_length_translation,
                    num_beams=4
                )
            
            batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(batch_translations)
            
        # Cleanup
        print("Cleaning up Translator model...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return translations

    def run(self):
        # Initialize W&B
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config.__dict__
        )
        
        # Load Data
        test_data = self.load_data()
        input_texts = test_data['latin_text']
        references = test_data['summary_de']
        
        # Step 1: Summarize
        latin_summaries = self.run_summarization(input_texts)
        
        # Step 2: Translate
        german_summaries = self.run_translation(latin_summaries)
        
        # Store results
        for i in range(len(input_texts)):
            self.results.append({
                "input_text": input_texts[i],
                "latin_summary": latin_summaries[i],
                "prediction": german_summaries[i],
                "reference": references[i]
            })
        
        # Evaluate
        print("\n=== Evaluation ===")
        evaluator = SummarizationEvaluator(
            bertscore_model=self.config.bertscore_model,
            bertscore_num_layers=12,
            mauve_model=self.config.mauve_model,
            verbose=True
        )
        
        # Get tokenizer for tokenizer metrics (use the translator's tokenizer)
        tokenizer_for_eval = self.translator_tokenizer
        
        metrics = evaluator.evaluate(
            predictions=german_summaries,
            references=references,
            inputs=input_texts,
            use_expert_rules=True,
            tokenizer_id_or_obj=tokenizer_for_eval
        )
        
        # Log to W&B
        wandb.log(metrics.to_dict())
        
        # Log Table
        table = wandb.Table(columns=["Input", "Latin Summary (Step 1)", "German Summary (Step 2)", "Reference"])
        for res in self.results:
            table.add_data(
                res["input_text"][:500],
                res["latin_summary"],
                res["prediction"],
                res["reference"]
            )
        wandb.log({"predictions": table})
        
        wandb.finish()
        print("Experiment finished successfully!")

def build_slurm_python_args(config: STTConfig) -> List[str]:
    """Builds Python arguments for SLURM submission"""
    args = [
        "--summarizer", config.summarizer_model_id,
        "--translator", config.translator_model_id,
        "--dataset", config.dataset_path,
        "--n_samples", str(config.n_test_samples),
        "--batch_size", str(config.batch_size),
        "--run_name", config.wandb_run_name,
        "--prompts_file", config.prompts_file,
        "--run_mode", "local"
    ]
    
    if config.prompt_version:
        args.extend(["--prompt_version", config.prompt_version])
        
    return args

def main():
    parser = argparse.ArgumentParser(description="STT Pipeline (Summarize-Then-Translate)")
    # Model args
    parser.add_argument("--summarizer", type=str, default="LatinNLP/LatinSummarizerModel")
    parser.add_argument("--translator", type=str, default="facebook/nllb-200-3.3B")
    parser.add_argument("--dataset", type=str, default="/dss/dsshome1/0E/ra42qov2/development/data/all.csv")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--run_name", type=str, default="stt-latin-nllb")
    
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
    
    config = STTConfig(
        summarizer_model_id=args.summarizer,
        translator_model_id=args.translator,
        dataset_path=args.dataset,
        n_test_samples=args.n_samples,
        batch_size=args.batch_size,
        wandb_run_name=args.run_name,
        prompts_file=args.prompts_file,
        prompt_version=args.prompt_version,
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
        job_name = args.slurm_job_name or f"stt-{args.run_name}"
        
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
        print(f"Summarizer: {config.summarizer_model_id}")
        print(f"Translator: {config.translator_model_id}")
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
        pipeline = STTPipeline(config)
        pipeline.run()

if __name__ == "__main__":
    main()
