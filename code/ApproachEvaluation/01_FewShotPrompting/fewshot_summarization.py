#!/usr/bin/env python3
# See the makefile for instructions on how to run this script

import os
import sys
import argparse
import subprocess
import tempfile
import torch
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Any
from abc import ABC, abstractmethod
from pathlib import Path
import time
import gc

import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    BitsAndBytesConfig
)
import google.generativeai as genai # for the Gemini API

# Import modules from src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluator import SummarizationEvaluator, EvaluationMetrics
from src.slurm_manager import SlurmConfig, SlurmJobSubmitter
from src.model_wrappers import BaseModelWrapper, CausalLMWrapper, Seq2SeqWrapper, APIModelWrapper

def clean_thinking_tokens(text: str) -> str:
    """
    Removes Qwen3's thinking tokens from generated text.
    Qwen3 uses <think>...</think> or just </think> markers.
    """
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</think>', '', text)
    text = re.sub(r'<think>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@dataclass
class ExperimentConfig:
    model_id: str
    model_type: str  # 'causal', 'seq2seq', or 'api'
    dataset_path: str
    n_shots: int = 5
    n_test_samples: int = 50
    max_new_tokens: int = 256
    max_context_length: int = 16384
    use_quantization: bool = False
    quantization_bits: int = 4
    bertscore_model: str = "magistermilitum/bert_medieval_multilingual"
    mauve_model: str = "magistermilitum/bert_medieval_multilingual"  # Medieval BERT for medieval Latin/German
    wandb_project: str = "crosslingual-summarization-thesis"
    wandb_job_type: str = "baseline-fewshot"
    wandb_run_name: Optional[str] = None
    max_memory_per_gpu: str = "78GB"
    use_chat_template: bool = True
    num_beams: int = 1
    do_sample: bool = False
    # API-specific parameters
    api_key_env_var: Optional[str] = None  # Name of the environment variable with API Key
    api_temperature: float = 0.0
    api_top_p: float = 0.95
    api_requests_per_minute: int = 10  # Rate limiting
    # Prompt Versioning
    system_prompt_version: Optional[str] = None  # Version of the System Prompt (None = latest)
    system_prompts_file: str = "system_prompts.json"  # Path to the prompt definitions file


class FewShotExperiment:
    """Main class for the Few-Shot Experiment"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_wrapper = self._create_model_wrapper()
        self.results = []
        
    def _create_model_wrapper(self) -> BaseModelWrapper:
        """Factory method for Model Wrapper"""
        if self.config.model_type == "seq2seq":
            return Seq2SeqWrapper(self.config)
        elif self.config.model_type == "causal":
            return CausalLMWrapper(self.config)
        elif self.config.model_type == "api":
            return APIModelWrapper(self.config)
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}. Supported: 'causal', 'seq2seq', 'api'")
    
    def load_and_prepare_data(self):
        """Loads and prepares the dataset - uses whitelist or filters blacklist"""
        print(f"Loading dataset: {self.config.dataset_path}")
        dataset = load_dataset("csv", data_files=self.config.dataset_path, engine="python")
        train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
        
        test_dataset = train_test_split['test']
        whitelisted_ids = self.model_wrapper.whitelisted_charter_ids
        blacklisted_ids = self.model_wrapper.blacklisted_charter_ids
        
        # Whitelist has priority
        if whitelisted_ids:
            print(f"Whitelist active: Filtering test set to {len(whitelisted_ids)} specific Charter IDs...")
            original_test_size = len(test_dataset)
            
            # Filter only the Charter IDs that are in the whitelist
            test_dataset = test_dataset.filter(
                lambda sample: str(sample['nummer']) in whitelisted_ids
            )
            
            selected_count = len(test_dataset)
            print(f"  → {selected_count} of {len(whitelisted_ids)} Charter(s) found in test set")
            
            if selected_count < len(whitelisted_ids):
                missing_ids = set(whitelisted_ids) - set(str(s['nummer']) for s in test_dataset)
                print(f"Warning: {len(missing_ids)} Charter ID(s) not found in dataset: {list(missing_ids)[:5]}")
            
            # With Whitelist: Use ALL found Charters (ignore n_test_samples)
            test_dataset_subset = test_dataset
            print(f"Evaluating on all {len(test_dataset_subset)} whitelisted Charters")
            
        elif blacklisted_ids:
            print(f"Filtering {len(blacklisted_ids)} blacklisted Charter IDs from test set...")
            original_test_size = len(test_dataset)
            
            # Filter Charters with blacklisted IDs
            test_dataset = test_dataset.filter(
                lambda sample: str(sample['nummer']) not in blacklisted_ids
            )
            
            filtered_count = original_test_size - len(test_dataset)
            if filtered_count > 0:
                print(f"{filtered_count} Charter(s) removed from test set (were in System Prompt)")
                print(f"Remaining test samples: {len(test_dataset)}")
            
            # Select test subset
            n_test = min(self.config.n_test_samples, len(test_dataset))
            test_dataset_subset = test_dataset.shuffle(seed=42).select(range(n_test))
            
        else:
            # No filters active
            n_test = min(self.config.n_test_samples, len(test_dataset))
            test_dataset_subset = test_dataset.shuffle(seed=42).select(range(n_test))
        
        print(f"\n Dataset prepared:")
        print(f"  Train set: {len(train_test_split['train'])} samples")
        print(f"  Test set (after filtering): {len(test_dataset)} samples")
        print(f"  Test samples used: {len(test_dataset_subset)} samples")
        
        return train_test_split['train'], test_dataset_subset
    
    def run_inference(self, train_data, test_data):
        """Performs inference on the test set"""
        print("Starting inference loop...")
        generation_config = self.model_wrapper.prepare_generation_config()
        
        for i, test_sample in enumerate(test_data):
            print(f"Processing sample {i+1}/{len(test_data)}...", end='\r')
            
            prompt, reference = self.model_wrapper.create_prompt(
                test_sample, train_data, self.config.n_shots
            )
            
            prediction = self.model_wrapper.generate(prompt, generation_config)
            
            # Clean thinking tokens (especially for Qwen3 models)
            prediction = clean_thinking_tokens(prediction)
            
            self.results.append({
                "prompt": prompt,
                "input_text": test_sample['latin_text'],
                "prediction": prediction,
                "reference": reference
            })
        
        print(f"\nInference completed for {len(self.results)} samples.")
    
    def compute_metrics(self) -> Dict[str, float]:
        """Computes evaluation metrics"""
        predictions = [r['prediction'] for r in self.results]
        references = [r['reference'] for r in self.results]
        inputs = [r['input_text'] for r in self.results]
        
        # Save tokenizer for analysis (if available)
        tokenizer_for_eval = None
        if hasattr(self.model_wrapper, 'tokenizer') and self.model_wrapper.tokenizer is not None:
            tokenizer_for_eval = self.model_wrapper.tokenizer
        
        # Free memory (unload current model) before evaluator starts
        # This is important as the evaluator internally loads the Judge model
        if hasattr(self.model_wrapper, 'model'):
            print("  Unloading current model to free VRAM for evaluator...")
            
            # Move model to CPU first to free GPU memory
            if hasattr(self.model_wrapper.model, 'to'):
                self.model_wrapper.model.to('cpu')
            
            # Delete model and tokenizer
            del self.model_wrapper.model
            if hasattr(self.model_wrapper, 'tokenizer'):
                del self.model_wrapper.tokenizer
            
            # Force garbage collection multiple times
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            
            # Reset CUDA stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            print(f"   Model moved to CPU and deleted")
            print(f"   CUDA cache cleared and synchronized")
        
        evaluator = SummarizationEvaluator(
            bertscore_model=self.config.bertscore_model,
            bertscore_num_layers=12,
            mauve_model=self.config.mauve_model,
            verbose=True
        )
        
        metrics: EvaluationMetrics = evaluator.evaluate(
            predictions=predictions,
            references=references,
            inputs=inputs,
            use_expert_rules=True,
            tokenizer_id_or_obj=tokenizer_for_eval
        )
        
        # Convert EvaluationMetrics to Dictionary for W&B
        return metrics.to_dict()
    
    def log_to_wandb(self, metrics: Dict[str, float]):
        """Logs results to Weights & Biases"""
        print("Logging results to W&B...")
        
        # Log metrics
        wandb.log(metrics)
        
        # Create example table
        results_table = wandb.Table(
            columns=["Prompt", "Input Text", "Prediction", "Reference (Ground Truth)"]
        )
        for res in self.results:
            results_table.add_data(
                res['prompt'],
                res['input_text'],
                res['prediction'],
                res['reference']
            )
        wandb.log({"evaluation_samples": results_table})
        
        print("Results successfully logged.")
    
    def run(self):
        """Runs the complete experiment"""
        wandb_name = self.config.wandb_run_name or f"fewshot-{self.config.model_id.split('/')[-1]}-{self.config.n_shots}shot"
        wandb_config = self.config.__dict__.copy()
        wandb_config['system_prompt_version_used'] = self.model_wrapper.system_prompt.split('\n')[0][:100]
        wandb_config['blacklisted_charter_ids'] = self.model_wrapper.blacklisted_charter_ids
        wandb_config['num_blacklisted_charters'] = len(self.model_wrapper.blacklisted_charter_ids)
        wandb_config['whitelisted_charter_ids'] = self.model_wrapper.whitelisted_charter_ids
        wandb_config['num_whitelisted_charters'] = len(self.model_wrapper.whitelisted_charter_ids)
        wandb_config['using_whitelist'] = len(self.model_wrapper.whitelisted_charter_ids) > 0
        
        run = wandb.init(
            project=self.config.wandb_project,
            job_type=self.config.wandb_job_type,
            name=wandb_name,
            config=wandb_config
        )
        
        # Show W&B Run URL
        print(f"\n{'='*60}")
        print(f"W&B Run URL: {run.get_url()}")
        print(f"{'='*60}\n")
        
        # Load model
        self.model_wrapper.load_model()
        
        # Load data
        train_data, test_data = self.load_and_prepare_data()
        
        self.run_inference(train_data, test_data)
        metrics = self.compute_metrics()
        
        # Log to W&B
        self.log_to_wandb(metrics)
        
        # End W&B
        wandb.finish()
        print("Experiment completed!")


def build_slurm_python_args(experiment_config: ExperimentConfig) -> List[str]:
    """
    Builds Python command-line arguments from ExperimentConfig for SLURM submission
    """
    args = [
        "--model_id", experiment_config.model_id,
        "--model_type", experiment_config.model_type,
        "--dataset_path", experiment_config.dataset_path,
        "--n_shots", str(experiment_config.n_shots),
        "--n_test_samples", str(experiment_config.n_test_samples),
        "--max_new_tokens", str(experiment_config.max_new_tokens),
        "--max_context_length", str(experiment_config.max_context_length),
        "--max_memory_per_gpu", experiment_config.max_memory_per_gpu,
        "--wandb_project", experiment_config.wandb_project,
        "--num_beams", str(experiment_config.num_beams),
    ]
    
    if experiment_config.use_quantization:
        args.append("--use_quantization")
        args.extend(["--quantization_bits", str(experiment_config.quantization_bits)])
    
    if experiment_config.wandb_run_name:
        args.extend(["--wandb_run_name", experiment_config.wandb_run_name])
    
    if not experiment_config.use_chat_template:
        args.append("--no_chat_template")
    
    if experiment_config.do_sample:
        args.append("--do_sample")
    
    # Prompt Versioning
    if experiment_config.system_prompt_version:
        args.extend(["--system_prompt_version", experiment_config.system_prompt_version])
    
    if experiment_config.system_prompts_file != "system_prompts.json":
        args.extend(["--system_prompts_file", experiment_config.system_prompts_file])
    
    # Evaluation-Parameter
    if experiment_config.mauve_model != "magistermilitum/bert_medieval_multilingual":
        args.extend(["--mauve_model", experiment_config.mauve_model])
    
    # API-specific parameters
    if experiment_config.api_key_env_var:
        args.extend(["--api_key_env_var", experiment_config.api_key_env_var])
        args.extend(["--api_temperature", str(experiment_config.api_temperature)])
        args.extend(["--api_top_p", str(experiment_config.api_top_p)])
        args.extend(["--api_requests_per_minute", str(experiment_config.api_requests_per_minute)])
    
    # Important: Add --run_mode local to prevent recursive submission
    args.extend(["--run_mode", "local"])
    
    return args


def parse_arguments():
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Few-Shot Summarization for medieval Latin texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # === Execution Mode ===
    parser.add_argument(
        "--run_mode",
        type=str,
        choices=["local", "slurm"],
        default="local",
        help="Execution mode: 'local' for local execution or 'slurm' for cluster submission (default: local)"
    )
    
    # === Model Arguments ===",
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace Model ID (e.g. 'meta-llama/Llama-4-Scout-17B-16E-Instruct')"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["causal", "seq2seq", "api"],
        default="causal",
        help="Type of model: 'causal' for LLMs, 'seq2seq' for T5-like models, or 'api' for API-based models"
    )
    
    # Data Arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/dss/dsshome1/0E/ra42qov2/development/data/all.csv",
        help="Path to CSV dataset"
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=5,
        help="Number of few-shot examples (default: 5)"
    )
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=50,
        help="Number of test samples (default: 50)"
    )
    
    # Generation Arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens (default: 256)"
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=16384,
        help="Maximum context length (default: 16384)"
    )
    
    # Quantization
    parser.add_argument(
        "--use_quantization",
        action="store_true",
        help="Enables quantization for large models"
    )
    parser.add_argument(
        "--quantization_bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="Number of bits for quantization (4 or 8, default: 4)"
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default="78GB",
        help="Maximum VRAM per GPU (default: 78GB)"
    )
    
    # W&B Arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="crosslingual-summarization-thesis",
        help="W&B Project-Name"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B Run Name (default: auto-generated)"
    )
    
    # Further Options
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disables Chat Template for Causal LMs"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for Beam Search (default: 1 = greedy)"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enables sampling instead of greedy decoding"
    )
    
    # Evaluation Arguments
    parser.add_argument(
        "--mauve_model",
        type=str,
        default="magistermilitum/bert_medieval_multilingual",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "bert-base-multilingual-cased", "magistermilitum/bert_medieval_multilingual"],
        help="Model for MAUVE calculation (default: magistermilitum/bert_medieval_multilingual). Supports GPT-2 and BERT models."
    )
    
    # === API-specific Arguments ===
    api_group = parser.add_argument_group('API Options', 'Options for API-based models (only with --model_type api)')
    
    api_group.add_argument(
        "--api_key_env_var",
        type=str,
        default="GEMINI_API_KEY",
        help="Name of the environment variable with the API Key (default: GEMINI_API_KEY)"
    )
    api_group.add_argument(
        "--api_temperature",
        type=float,
        default=0.0,
        help="Temperature for API requests (default: 0.0 for deterministic)"
    )
    api_group.add_argument(
        "--api_top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus sampling) for API requests (default: 0.95)"
    )
    api_group.add_argument(
        "--api_requests_per_minute",
        type=int,
        default=10,
        help="Maximum number of API requests per minute (Rate Limiting, default: 10)"
    )
    
    # === Prompt Versioning ===
    prompt_group = parser.add_argument_group('Prompt Versioning', 'Options for System Prompt Management')
    
    prompt_group.add_argument(
        "--system_prompt_version",
        type=str,
        default=None,
        help="Version of the System Prompt (e.g. '1.0', '2.0'). If not specified, the latest version will be used."
    )
    prompt_group.add_argument(
        "--system_prompts_file",
        type=str,
        default="system_prompts.json",
        help="Path to the JSON file with System Prompt definitions (default: system_prompts.json in the same directory)"
    )
    
    # === SLURM-specific Arguments ===
    slurm_group = parser.add_argument_group('SLURM Options', 'Options for SLURM Cluster Submission (only with --run_mode slurm)')
    
    slurm_group.add_argument(
        "--slurm_partition",
        type=str,
        default="mcml-dgx-a100-40x8",
        help="SLURM Partition (default: mcml-dgx-a100-40x8)"
    )
    slurm_group.add_argument(
        "--slurm_gpus",
        type=int,
        default=1,
        help="Number of GPUs (default: 2)"
    )
    slurm_group.add_argument(
        "--slurm_time",
        type=str,
        default="04:00:00",
        help="Time limit in format HH:MM:SS (default: 04:00:00)"
    )
    slurm_group.add_argument(
        "--slurm_job_name",
        type=str,
        default=None,
        help="Job name (default: auto-generated from model name)"
    )
    slurm_group.add_argument(
        "--slurm_log_dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)"
    )
    slurm_group.add_argument(
        "--slurm_conda_env",
        type=str,
        default="llm-thesis",
        help="Conda environment name (default: llm-thesis)"
    )
    slurm_group.add_argument(
        "--slurm_email",
        type=str,
        default=None,
        help="Email for job notifications (default: none)"
    )
    slurm_group.add_argument(
        "--slurm_email_type",
        type=str,
        default="END,FAIL",
        help="Email notification types (default: END,FAIL)"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create Experiment Config from arguments
    experiment_config = ExperimentConfig(
        model_id=args.model_id,
        model_type=args.model_type,
        dataset_path=args.dataset_path,
        n_shots=args.n_shots,
        n_test_samples=args.n_test_samples,
        max_new_tokens=args.max_new_tokens,
        max_context_length=args.max_context_length,
        use_quantization=args.use_quantization,
        quantization_bits=args.quantization_bits,
        max_memory_per_gpu=args.max_memory_per_gpu,
        mauve_model=args.mauve_model,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_chat_template=not args.no_chat_template,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        # API-specific parameters
        api_key_env_var=args.api_key_env_var if args.model_type == "api" else None,
        api_temperature=args.api_temperature if args.model_type == "api" else 0.0,
        api_top_p=args.api_top_p if args.model_type == "api" else 0.95,
        api_requests_per_minute=args.api_requests_per_minute if args.model_type == "api" else 10,
        # Prompt Versioning
        system_prompt_version=args.system_prompt_version,
        system_prompts_file=args.system_prompts_file
    )
    
    # Decide based on run_mode
    if args.run_mode == "slurm":
        # Create SLURM Config
        # Auto-generate job name if not specified
        job_name = args.slurm_job_name or f"fewshot-{args.model_id.split('/')[-1]}-{args.n_shots}shot"
        
        slurm_config = SlurmConfig(
            partition=args.slurm_partition,
            num_gpus=args.slurm_gpus,
            time_limit=args.slurm_time,
            job_name=job_name,
            log_dir=args.slurm_log_dir,
            conda_env=args.slurm_conda_env,
            email=args.slurm_email,
            email_type=args.slurm_email_type
        )
        
        # Build Python arguments for SLURM submission
        python_args = build_slurm_python_args(experiment_config)
        
        # Submit Job to SLURM using the new slurm_manager
        print("=== SLURM Job Submission ===")
        print(f"Model: {experiment_config.model_id}")
        print(f"N-Shots: {experiment_config.n_shots}")
        print(f"GPUs: {slurm_config.num_gpus}")
        print(f"Time Limit: {slurm_config.time_limit}")
        print()
        
        submitter = SlurmJobSubmitter(
            script_path=str(Path(__file__).resolve()),
            python_args=python_args,
            slurm_config=slurm_config
        )
        job_id = submitter.submit()
        
    else:  # local
        # Run experiment locally
        print("=== Local Execution ===")
        print(f"Model: {experiment_config.model_id}")
        print(f"N-Shots: {experiment_config.n_shots}")
        print()
        
        experiment = FewShotExperiment(experiment_config)
        experiment.run()


if __name__ == "__main__":
    main()
