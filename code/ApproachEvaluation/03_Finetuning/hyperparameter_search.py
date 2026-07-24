#!/usr/bin/env python3
"""
Hyperparameter Search for Qwen3-0.6B Fine-tuning
"""

import argparse
import random
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime


class HyperparameterSearch:
    def __init__(
        self,
        search_type: str = "random",
        n_trials: int = 10,
        base_output_dir: str = "/dss/mcmlscratch/0E/ra42qov2/models/qwen3-0.6b-hp-search",
        slurm_partition: str = "mcml-hgx-a100-80x4",
        slurm_gpus: int = 1,
        slurm_time: str = "10:00:00",
        delay_between_jobs: int = 30
    ):
        self.search_type = search_type
        self.n_trials = n_trials
        self.base_output_dir = Path(base_output_dir)
        self.slurm_partition = slurm_partition
        self.slurm_gpus = slurm_gpus
        self.slurm_time = slurm_time
        self.delay_between_jobs = delay_between_jobs
        
        # Create output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track submitted jobs
        self.results_file = Path(__file__).parent / f"hp_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.submitted_jobs = []
    
    def define_search_space(self) -> Dict[str, Any]:
        search_space = {
            "learning_rate": [1e-4, 1.5e-4, 2e-4, 2.5e-4, 3e-4],
            "warmup_steps": [30, 40, 50, 60, 70],
            "weight_decay": [0.01, 0.02, 0.05, 0.1],
            "lora_r": [8, 16, 32],
            "batch_size": [2, 4, 8],
            "gradient_accumulation_steps": [2, 4, 8],
            "label_smoothing": [0.0, 0.05, 0.1, 0.15],
            "max_grad_norm": [0.3, 0.5, 1.0]
        }
        
        return search_space
    
    def sample_hyperparameters(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        config = {}
        
        for param, values in search_space.items():
            config[param] = random.choice(values)
        
        effective_batch_size = config["batch_size"] * config["gradient_accumulation_steps"]
        if effective_batch_size < 4:
            config["gradient_accumulation_steps"] = max(2, 4 // config["batch_size"])
        elif effective_batch_size > 16:
            config["gradient_accumulation_steps"] = max(1, 16 // config["batch_size"])
        
        # Set lora_alpha = 2 * lora_r (common QLoRA convention)
        config["lora_alpha"] = config["lora_r"] * 2
        
        return config
    
    def generate_grid_configs(self, search_space: Dict[str, Any], n_samples: int) -> List[Dict[str, Any]]:
        configs = []
        
        for _ in range(n_samples):
            config = self.sample_hyperparameters(search_space)
            configs.append(config)
        
        return configs
    
    def submit_training_job(self, config: Dict[str, Any], trial_id: int):
        run_name = (
            f"qwen3-0.6b-hp-trial{trial_id:03d}_"
            f"e{config['epochs']}_"
            f"lr{config['learning_rate']:.0e}_"
            f"wd{config['weight_decay']:.0e}_"
            f"r{config['lora_r']}_a{config['lora_alpha']}"
        )
        
        output_dir = self.base_output_dir / f"trial_{trial_id:03d}"
        
        # Build the command which is then used to start the actual training job
        cmd = [
            "python", "end2end_finetuning.py",
            "--model_id", "Qwen/Qwen3-0.6B",
            "--dataset", "../data/all.csv",
            "--output_dir", str(output_dir),
            "--epochs", str(config["epochs"]),
            "--batch_size", str(config["batch_size"]),
            "--gradient_accumulation_steps", str(config["gradient_accumulation_steps"]),
            "--learning_rate", str(config["learning_rate"]),
            "--warmup_steps", str(config["warmup_steps"]),
            "--weight_decay", str(config["weight_decay"]),
            "--max_grad_norm", str(config["max_grad_norm"]),
            "--label_smoothing", str(config["label_smoothing"]),
            "--use_qlora",
            "--lora_r", str(config["lora_r"]),
            "--lora_alpha", str(config["lora_alpha"]),
            "--use_expert_rules",
            "--run_mode", "slurm",
            "--slurm_partition", self.slurm_partition,
            "--slurm_gpus", str(self.slurm_gpus),
            "--slurm_time", self.slurm_time,
            "--slurm_job_name", f"hp-trial-{trial_id:03d}",
            "--run_name", run_name
        ]
        
        print(f"\n{'='*60}")
        print(f"Trial {trial_id}/{self.n_trials}")
        print(f"{'='*60}")
        print(f"Configuration:")
        for key, value in config.items():
            print(f"  {key:30s}: {value}")
        print(f"\nSubmitting job: {run_name}")
        
        # Submit job
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  Job submitted successfully")
            print(f"  Output directory: {output_dir}")
            
            # Try to extract job ID from output
            job_id = None
            for line in result.stdout.split('\n'):
                if 'Submitted batch job' in line:
                    job_id = line.split()[-1]
                    print(f"  SLURM Job ID: {job_id}")
            
            # Record job info
            job_info = {
                "trial_id": trial_id,
                "job_id": job_id,
                "run_name": run_name,
                "config": config,
                "submitted_at": datetime.now().isoformat(),
                "output_dir": str(output_dir)
            }
            self.submitted_jobs.append(job_info)
            
        else:
            print(f"Job submission failed...")
            print(f"Error: {result.stderr}")
        
        # Save progress
        self.save_results()
        
        return result.returncode == 0
    
    def save_results(self):
        with open(self.results_file, 'w') as f:
            json.dump(self.submitted_jobs, f, indent=2)
    
    def run_search(self):
        print("="*60)
        print("Hyperparameter Search for Qwen3-0.6B Fine-tuning")
        print("="*60)
        print(f"Search type: {self.search_type}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Base output directory: {self.base_output_dir}")
        print(f"SLURM partition: {self.slurm_partition}")
        print(f"Results will be saved to: {self.results_file}")
        print("="*60)
        
        # Define search space
        search_space = self.define_search_space()
        
        print("\nSearch space:")
        for param, values in search_space.items():
            print(f"  {param:30s}: {values}")
        
        # Generate configurations
        if self.search_type == "random":
            configs = [self.sample_hyperparameters(search_space) for _ in range(self.n_trials)]
        elif self.search_type == "grid":
            configs = self.generate_grid_configs(search_space, self.n_trials)
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")
        
        # Set epochs to fixed value for all configs
        for config in configs:
            config["epochs"] = 10
        
        # Submit jobs
        successful_submissions = 0
        for i, config in enumerate(configs, 1):
            if self.submit_training_job(config, i):
                successful_submissions += 1
            
            # Delay between submissions to avoid overwhelming the scheduler
            if i < len(configs):
                print(f"\nWaiting {self.delay_between_jobs}s before next submission...")
                time.sleep(self.delay_between_jobs)
        
        print("\n" + "="*60)
        print("Hyperparameter Search Submission Complete")
        print("="*60)
        print(f"Total trials: {self.n_trials}")
        print(f"Successful submissions: {successful_submissions}")
        print(f"Failed submissions: {self.n_trials - successful_submissions}")
        print(f"\nResults file: {self.results_file}")
        print("\nMonitor jobs with: squeue -u $USER")
        print("View results in W&B: https://wandb.ai")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for Qwen3-0.6B fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--search_type",
        choices=["random", "grid"],
        default="random",
        help="Type of search (default: random)"
    )
    
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of trials to run (default: 10)"
    )
    
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="/dss/mcmlscratch/0E/ra42qov2/models/qwen3-0.6b-hp-search",
        help="Base directory for trial outputs"
    )
    
    parser.add_argument(
        "--slurm_partition",
        type=str,
        default="mcml-hgx-a100-80x4",
        help="SLURM partition (default: mcml-hgx-a100-80x4)"
    )
    
    parser.add_argument(
        "--slurm_gpus",
        type=int,
        default=1,
        help="Number of GPUs per job (default: 1)"
    )
    
    parser.add_argument(
        "--slurm_time",
        type=str,
        default="10:00:00",
        help="Time limit per job (default: 10:00:00)"
    )
    
    parser.add_argument(
        "--delay",
        type=int,
        default=30,
        help="Delay in seconds between job submissions (default: 30)"
    )
    
    args = parser.parse_args()
    
    search = HyperparameterSearch(
        search_type=args.search_type,
        n_trials=args.n_trials,
        base_output_dir=args.base_output_dir,
        slurm_partition=args.slurm_partition,
        slurm_gpus=args.slurm_gpus,
        slurm_time=args.slurm_time,
        delay_between_jobs=args.delay
    )
    
    search.run_search()


if __name__ == "__main__":
    main()
