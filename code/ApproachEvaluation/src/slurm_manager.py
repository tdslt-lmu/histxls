#!/usr/bin/env python3
"""
SLURM Job Submission Manager
"""

import sys
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class SlurmConfig:
    """Configuration for SLURM job submission"""
    partition: str = "mcml-dgx-a100-40x8"
    num_gpus: int = 2
    time_limit: str = "04:00:00"
    job_name: str = "job"
    log_dir: str = "logs"
    conda_env: Optional[str] = None
    email: Optional[str] = None
    email_type: str = "END,FAIL"
    qos: Optional[str] = None
    
    def __post_init__(self):
        if self.qos is None:
            if "mcml" in self.partition.lower():
                self.qos = "mcml"


class SlurmJobSubmitter:
    def __init__(
        self,
        script_path: str,
        python_args: List[str],
        slurm_config: SlurmConfig,
        hf_cache_path: Optional[str] = None,
        additional_setup: Optional[str] = None
    ):
        self.script_path = Path(script_path).resolve()
        self.python_args = python_args
        self.slurm_config = slurm_config
        self.hf_cache_path = hf_cache_path or "/dss/mcmlscratch/0E/ra42qov2/hf_cache"
        self.additional_setup = additional_setup
        
        if not self.script_path.exists():
            raise FileNotFoundError(f"Script not found: {self.script_path}")
    
    def create_sbatch_script(self) -> str:
        # Create log directory if it doesn't exist
        log_dir = Path(self.slurm_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Build sbatch header
        script = self._build_sbatch_header()
        
        # Add environment setup
        script += self._build_environment_setup()
        
        # Add GPU info
        script += self._build_gpu_info()
        
        # Add experiment info
        script += self._build_experiment_info()
        
        # Add additional setup if provided
        if self.additional_setup:
            script += f"\n{self.additional_setup}\n\n"
        
        # Add Python command
        script += self._build_python_command()
        
        # Add completion message
        script += '\necho ""\necho "=== Experiment Completed ==="\n'
        
        return script
    
    def _build_sbatch_header(self) -> str:
        script = f"""#!/bin/bash
#SBATCH --job-name={self.slurm_config.job_name}
#SBATCH --output={self.slurm_config.log_dir}/{self.slurm_config.job_name}_%j.out
#SBATCH --error={self.slurm_config.log_dir}/{self.slurm_config.job_name}_%j.err
#SBATCH --time={self.slurm_config.time_limit}
#SBATCH --partition={self.slurm_config.partition}
#SBATCH --gres=gpu:{self.slurm_config.num_gpus}
"""
        
        # Add QoS if specified
        if self.slurm_config.qos:
            script += f"#SBATCH --qos={self.slurm_config.qos}\n"
        
        script = script.rstrip() + "\n"
        
        # Add email notifications if specified
        if self.slurm_config.email:
            script += f"#SBATCH --mail-user={self.slurm_config.email}\n"
            script += f"#SBATCH --mail-type={self.slurm_config.email_type}\n"
        
        script += "\n"
        return script
    
    def _build_environment_setup(self) -> str:
        if not self.slurm_config.conda_env:
            return ""
        
        script = f"""
# Aktiviere Conda Environment
set +u
source ~/.zshrc
eval "$(conda shell.zsh hook)"
mamba activate {self.slurm_config.conda_env}
set -u

# Verify correct environment
echo "=== Python Environment ==="
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Setze HuggingFace Cache
export HF_HOME="{self.hf_cache_path}"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p $HF_HOME
echo "HuggingFace Cache: $HF_HOME"

# Kopiere HF-Token falls vorhanden
DEFAULT_TOKEN_PATH="$HOME/.cache/huggingface/token"
NEW_TOKEN_PATH="$HF_HOME/token"
if [ -f "$DEFAULT_TOKEN_PATH" ]; then
    cp "$DEFAULT_TOKEN_PATH" "$NEW_TOKEN_PATH"
fi

export TOKENIZERS_PARALLELISM=false

"""
        return script
    
    def _build_gpu_info(self) -> str:
        return """
# Zeige GPU Information
echo "=== GPU Information ==="
nvidia-smi
echo ""

"""
    
    def _build_experiment_info(self) -> str:
        # Extract some common arguments for display
        info_args = {}
        i = 0
        while i < len(self.python_args):
            arg = self.python_args[i]
            if arg.startswith("--"):
                key = arg[2:]
                if i + 1 < len(self.python_args) and not self.python_args[i + 1].startswith("--"):
                    info_args[key] = self.python_args[i + 1]
                    i += 2
                else:
                    info_args[key] = "True"
                    i += 1
            else:
                i += 1
        
        script = """
# Zeige Experiment-Konfiguration
echo "=== Experiment Configuration ==="
"""
        
        # Display common parameters
        for key in ["model_id", "pipeline_mode", "n_shots", "max_context_length", "use_quantization"]:
            if key in info_args:
                display_key = key.replace("_", " ").title()
                script += f'echo "{display_key}: {info_args[key]}"\n'
        
        script += 'echo ""\n\n'
        return script
    
    def _build_python_command(self) -> str:
        # Build command with proper line continuation
        cmd_parts = ["python", str(self.script_path)] + self.python_args
        
        # Format with line continuations for readability
        script = """
# Führe Experiment aus
echo "=== Starting Experiment ==="
"""
        script += " \\\n    ".join(cmd_parts) + "\n"
        return script
    
    def submit(self) -> str:
        # Create temporary sbatch script
        script_content = self.create_sbatch_script()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sbatch', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # Submit the job
            result = subprocess.run(
                ['sbatch', script_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract Job-ID from output (e.g., "Submitted batch job 12345")
            output = result.stdout.strip()
            job_id = output.split()[-1]
            
            # Print success message
            self._print_success_message(job_id)
            
            return job_id
            
        except subprocess.CalledProcessError as e:
            print(f"Fehler beim Submitten des Jobs:", file=sys.stderr)
            print(f"  {e.stderr}", file=sys.stderr)
            sys.exit(1)
        finally:
            # Delete temporary script
            Path(script_path).unlink(missing_ok=True)
    
    def _print_success_message(self, job_id: str):
        print(f" Job erfolgreich submitted: {job_id}")
        print(f"  Logs: {self.slurm_config.log_dir}/{self.slurm_config.job_name}_{job_id}.{{out,err}}")
        print(f"\nÜberwache Status mit: squeue -j {job_id}")
        print(f"Logs anzeigen mit: tail -f {self.slurm_config.log_dir}/{self.slurm_config.job_name}_{job_id}.out")


def build_python_args(
    script_path: str,
    config_dict: Dict[str, Any],
    exclude_keys: Optional[List[str]] = None,
    flag_keys: Optional[List[str]] = None
) -> List[str]:
    exclude_keys = exclude_keys or []
    flag_keys = flag_keys or []
    
    args = []
    for key, value in config_dict.items():
        if key in exclude_keys:
            continue
        
        arg_name = f"--{key}"
        
        if key in flag_keys:
            # Boolean flag - only add if True
            if value:
                args.append(arg_name)
        else:
            # Regular argument with value
            args.extend([arg_name, str(value)])
    
    return args
