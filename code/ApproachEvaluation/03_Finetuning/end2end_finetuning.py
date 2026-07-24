import os
import sys
import argparse
import torch
import gc
import wandb
import pandas as pd
import tempfile

# Redirect temp directory to scratch space to avoid /tmp/ filling up
# This is critical for W&B table logging which creates large temp files
os.environ['TMPDIR'] = os.environ.get('MCMLSCRATCH', '/tmp')
tempfile.tempdir = os.environ['TMPDIR']
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    T5Tokenizer,  # For MADLAD
    T5ForConditionalGeneration,  # For MADLAD
    Seq2SeqTrainingArguments,
    TrainingArguments,
    Seq2SeqTrainer,
    Trainer,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from transformers.optimization import Adafactor  # For mT5 optimizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# This here is needed so that I call import the classes of the src folder
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluator import SummarizationEvaluator, EvaluationMetrics
from src.slurm_manager import SlurmConfig, SlurmJobSubmitter


@dataclass
class FinetuningConfig:
    model_id: str
    dataset_path: str
    output_dir: str
    
    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-6
    warmup_steps: int = 10
    weight_decay: float = 0.1
    max_grad_norm: float = 0.5
    label_smoothing: float = 0.1
    max_input_length: int = 2048  # the latin texts can be quite long, so I have to set it rather long.
    max_target_length: int = 256
    
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Evaluation
    eval_only: bool = False
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 20
    use_expert_rules: bool = False  # Disabled by default, as its expensive (needs to load Qwen3-8B)
    
    # QLoRA options
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # I have a if-clause in the code later where I define default target modules based on model type
    
    # Optimizer options
    optimizer: str = "auto"  # "auto" (default AdamW, Adafactor for mT5), "adamw", "adafactor"
    
    wandb_project: str = "crosslingual-summarization-thesis"
    wandb_run_name: str = "end2end-finetuning"
    
    debug_mode: bool = False
    debug_sample_ratio: float = 0.1
    
    run_mode: str = "local"
    slurm_partition: str = "mcml-dgx-a100-40x8"
    slurm_gpus: int = 1
    slurm_time: str = "06:00:00"
    slurm_job_name: Optional[str] = None
    slurm_log_dir: str = "logs"
    slurm_conda_env: str = "llm-thesis"
    slurm_email: Optional[str] = None


class End2EndFinetuner:
    def __init__(self, config: FinetuningConfig):
        # Setting some default class variables which I need to access later...
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_encoder_decoder = None  # Will be determined when loading model
        self.results_csv_path = Path(__file__).parent / "results.csv"
        self.wandb_run_url = None
        self.num_model_parameters = None
        
        print(f"   End-to-End Latin→German Summarization Finetuning")
        print(f"   Model: {config.model_id}")
        print(f"   Device: {self.device}")
        print(f"   Output: {config.output_dir}")
        print()
    
    def load_and_split_data(self) -> DatasetDict:
        """Loads CSV and creates train/val/test splits"""
        print("Loading dataset...")
        df = pd.read_csv(self.config.dataset_path)
        
        print(f"   Total samples: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Filter out None/empty values
        df = df.dropna(subset=['latin_text', 'summary_de'])
        df = df[df['latin_text'].str.strip() != ""]
        df = df[df['summary_de'].str.strip() != ""]
        print(f"   After filtering: {len(df)} samples")
        
        # Create Dataset from pandas
        dataset = Dataset.from_pandas(df)
        
        # If the debug mode is enabled, then I only want to use like 10% of the data the test faster
        if self.config.debug_mode:
            debug_size = int(len(dataset) * self.config.debug_sample_ratio)
            dataset = dataset.select(range(debug_size))
            print(f"   DEBUG MODE: Using only {debug_size} samples ({self.config.debug_sample_ratio:.0%})")
        
        train_size = int(self.config.train_split * len(dataset))
        val_size = int(self.config.val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        print(f"\n Data splits:")
        print(f"   Train: {train_size} ({self.config.train_split:.0%})")
        print(f"   Val:   {val_size} ({self.config.val_split:.0%})")
        print(f"   Test:  {test_size} ({self.config.test_split:.0%})")
        dataset = dataset.shuffle(seed=42)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, len(dataset)))
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        })
        
        return dataset_dict
    
    def preprocess_function_seq2seq(self, examples):
        # Add task prefix for T5-like models (T5, mT5, LongT5, etc.)
        model_id_lower = self.config.model_id.lower()
        
        if "madlad" in model_id_lower:
            task_prefix = "<2de> "  # This special prefix is required by the madlad model in order to output a specific language
        elif "philta" in model_id_lower or "bowphs" in model_id_lower:
            task_prefix = ""  # PhilTa doesn't use a prefix - it's trained on raw Latin/Greek/English text
        elif "mt5" in model_id_lower:
            task_prefix = ""  # mT5 doesn't use a prefix - it was pre-trained only on span corruption
        elif any(name in model_id_lower for name in ["t5", "longt5"]):
            task_prefix = "summarize: " # needed for standard T5 and LongT5
        elif any(name in model_id_lower for name in ["bart", "nllb"]):
            task_prefix = "" # no special prefix known to me for those models...
        else:
            task_prefix = ""
        
        inputs = [task_prefix + text for text in examples['latin_text']]
        targets = examples['summary_de']
        
        # Now I have to tokenize both the input and the output lables...
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_input_length,
            truncation=True,
            padding=False  # Padding handled by data collator
        )
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.config.max_target_length,
            truncation=True,
            padding=False
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def preprocess_function_causal(self, examples):
        # I will use for the beginning a simple format: Latin input + German summary
        texts = []
        for latin, summary in zip(examples['latin_text'], examples['summary_de']):
            # Format: {latin}\n\n{summary}<eos>
            text = f"{latin}\n\n{summary}{self.tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize the texts WITHOUT truncation (else it may cut off very important information...)
        model_inputs = self.tokenizer(
            texts,
            truncation=False,
            padding=False
        )
        
        # The problem with causal LLMs is that they consider the entire sequence (Latin input + German summary) as a single continuous text document and would by default compute a loss for every token
        # --> The model learns to reproduce the Latin input as well instead of only generating the summary --> I don't want that
        # Therefore, the labels for all tokens of the Latin input are masked with -100, so that PyTorch ignores them during loss calculation and the model is trained exclusively on generating the German Kopfregest!
        labels = []
        for i, (latin, summary) in enumerate(zip(examples['latin_text'], examples['summary_de'])):
            latin_with_sep = f"{latin}\n\n"
            latin_ids = self.tokenizer(latin_with_sep, add_special_tokens=False)["input_ids"]
            full_ids = model_inputs["input_ids"][i]
            
            # Using -100 for Latin input (ignored according to PyTorch loss calculation)
            label_ids = [-100] * len(latin_ids) + full_ids[len(latin_ids):]
            
            label_ids = label_ids[:len(full_ids)]
            if len(label_ids) < len(full_ids):
                label_ids.extend(full_ids[len(label_ids):])
            
            labels.append(label_ids)
        
        model_inputs["labels"] = labels
        
        return model_inputs
    
    def prepare_datasets(self, dataset_dict: DatasetDict):
        print("\n Tokenizing datasets...")
        
        # I have to distinguish between encoder-decoder models like T5 or decoder-only models like Llama or Qwen
        if self.is_encoder_decoder:
            preprocess_fn = self.preprocess_function_seq2seq
        else:
            preprocess_fn = self.preprocess_function_causal
        
        tokenized_datasets = dataset_dict.map(
            preprocess_fn,
            batched=True,
            remove_columns=dataset_dict['train'].column_names,
            desc="Tokenizing"
        )
        
        print(f"   Tokenization complete")
        return tokenized_datasets
    
    def train(self):
        # Weights and Bias initialization
        wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config=self.config.__dict__
        )
        self.wandb_run_url = wandb_run.get_url()
        
        dataset_dict = self.load_and_split_data()
        
        print(f"\nLoading model: {self.config.model_id}")
        
        # CRITICAL: MADLAD requires T5Tokenizer, not AutoTokenizer
        model_id_lower = self.config.model_id.lower()
        if "madlad" in model_id_lower:
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_id)
            print(f"   T5Tokenizer loaded (required for MADLAD)")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, trust_remote_code=True)
        
        # Is needed for decoder only models like Qwen...
        if hasattr(self.tokenizer, 'padding_side'):
            self.tokenizer.padding_side = 'right'
        
        if self.config.use_qlora:
            print("   Loading with QLoRA (4-bit quantization)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, # matrix mult in forward pass is faster in float16
                bnb_4bit_use_double_quant=True
            )
            quantization_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        else:
            quantization_kwargs = {}
        
        # Distinguish between encoder-decoder and decoder-only models
        print("   Detecting model architecture...")
        
        # CRITICAL: MADLAD requires T5ForConditionalGeneration, not AutoModelForSeq2SeqLM
        model_id_lower = self.config.model_id.lower()
        
        try:
            if "madlad" in model_id_lower:
                # Load MADLAD with specific T5 class
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.config.model_id,
                    torch_dtype=torch.bfloat16,  # MADLAD uses bfloat16
                    device_map="auto"
                )
                # Enable gradient checkpointing for MADLAD
                if not self.config.use_qlora:
                    self.model.gradient_checkpointing_enable()
                    print(f"   Gradient checkpointing enabled for MADLAD")
                print(f"   Loaded MADLAD with T5ForConditionalGeneration (bfloat16)")
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_id, 
                    trust_remote_code=True,
                    **quantization_kwargs
                )
                print(f"   Loaded as Encoder-Decoder model (T5/BART/mT5/NLLB/LongT5/PhilTa/etc.)")
            
            self.is_encoder_decoder = True
            model_type = self.model.config.model_type if hasattr(self.model.config, 'model_type') else 'encoder-decoder'
            
            # CRITICAL FIX for PhilTa and mT5: Set decoder_start_token_id
            if "philta" in model_id_lower or "bowphs" in model_id_lower:
                self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
                print(f"   PhilTa detected: Set decoder_start_token_id = {self.tokenizer.pad_token_id}")
                
                # Enable gradient checkpointing for memory efficiency (when not using QLoRA)
                if not self.config.use_qlora:
                    self.model.gradient_checkpointing_enable()
                    print(f"   Gradient checkpointing enabled for memory efficiency")
            elif "mt5" in model_id_lower:
                self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
                print(f"   mT5 detected: Set decoder_start_token_id = {self.tokenizer.pad_token_id}")
                
                # Enable gradient checkpointing for memory efficiency (when not using QLoRA)
                if not self.config.use_qlora:
                    self.model.gradient_checkpointing_enable()
                    print(f"   Gradient checkpointing enabled for memory efficiency")
        except Exception as e:
            print(f"   Not an encoder-decoder model, trying causal LM...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    trust_remote_code=True,
                    **quantization_kwargs
                )
                self.is_encoder_decoder = False
                model_type = self.model.config.model_type if hasattr(self.model.config, 'model_type') else 'causal'
                print(f"   Loaded as Decoder-only model ({model_type}: Llama/Qwen/Mistral/etc.)")
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
                
                self.tokenizer.padding_side = 'left'
            except Exception as e2:
                raise ValueError(f"Could not load model as encoder-decoder or causal LM. Errors:\n1. Seq2Seq: {e}\n2. Causal: {e2}")
        
        # Different models have different layers to use QLoRA on...
        if self.config.use_qlora:
            print("   Applying LoRA adapters...")
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Auto-detect target modules if not specified
            if self.config.lora_target_modules is None:
                model_id_lower = self.config.model_id.lower()
                
                # Decoder-only models (Qwen, Llama, etc.)
                if any(name in model_id_lower for name in ["qwen", "llama", "mistral", "phi"]):
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                    print(f"   Using decoder-only LoRA targets: {target_modules}")
                
                # Encoder-decoder models (T5, mT5, BART, NLLB, LongT5, PhilTa, etc.)
                elif any(name in model_id_lower for name in ["t5", "mt5", "bart", "nllb", "longt5", "philta", "bowphs"]):
                    target_modules = ["q", "v"]  # Standard for T5-family (including PhilTa which is T5-based)
                    print(f"   Using encoder-decoder LoRA targets: {target_modules}")
                
                else:
                    target_modules = ["q", "v"] if self.is_encoder_decoder else ["q_proj", "v_proj"]
                    print(f"   Using generic LoRA targets: {target_modules}")
            else:
                target_modules = self.config.lora_target_modules
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM if not self.is_encoder_decoder else TaskType.SEQ_2_SEQ_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Logging the paramter count...
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {self.num_model_parameters:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        tokenized_datasets = self.prepare_datasets(dataset_dict)
        
        if self.is_encoder_decoder:
            # MADLAD doesn't use label_pad_token_id=-100, others do
            model_id_lower = self.config.model_id.lower()
            if "madlad" in model_id_lower:
                data_collator = DataCollatorForSeq2Seq(
                    tokenizer=self.tokenizer,
                    model=self.model,
                    padding=True
                )
            else:
                data_collator = DataCollatorForSeq2Seq(
                    tokenizer=self.tokenizer,
                    model=self.model,
                    padding=True,
                    label_pad_token_id=-100  # CRITICAL: Ignore pad tokens in loss calculation
                )
        else:
            # For causal LM with custom labels, use DataCollatorForSeq2Seq -> it handles padding for both input_ids and labels correctly
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True,
                label_pad_token_id=-100  # Pad labels with -100 (ignored in loss)
            )
        
        if self.is_encoder_decoder:
            # Determine if we should use bf16 (better for T5-family models like PhilTa, mT5, and MADLAD)
            model_id_lower = self.config.model_id.lower()
            use_bf16 = ("philta" in model_id_lower or "bowphs" in model_id_lower or "mt5" in model_id_lower or "madlad" in model_id_lower)
            use_fp16 = torch.cuda.is_available() and not use_bf16
            use_gradient_checkpointing = use_bf16 and not self.config.use_qlora  # Enable for PhilTa, mT5 and MADLAD when not using QLoRA
            
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.config.output_dir,
                eval_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_strategy="steps",
                save_steps=self.config.save_steps,
                logging_steps=self.config.logging_steps,
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                weight_decay=self.config.weight_decay,
                warmup_steps=self.config.warmup_steps,
                num_train_epochs=self.config.epochs,
                predict_with_generate=True,
                generation_max_length=self.config.max_target_length,
                fp16=use_fp16,
                bf16=use_bf16,  # Use bfloat16 for PhilTa (more stable than fp16 for T5)
                gradient_checkpointing=use_gradient_checkpointing,  # Enable for PhilTa to reduce memory
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                report_to="wandb",
                run_name=self.config.wandb_run_name,
                push_to_hub=False,
                max_grad_norm=self.config.max_grad_norm,
                label_smoothing_factor=self.config.label_smoothing
            )
        else:
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                eval_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_strategy="steps",
                save_steps=self.config.save_steps,
                logging_steps=self.config.logging_steps,
                learning_rate=self.config.learning_rate,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                weight_decay=self.config.weight_decay,
                warmup_steps=self.config.warmup_steps,
                num_train_epochs=self.config.epochs,
                fp16=torch.cuda.is_available(),
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                report_to="wandb",
                run_name=self.config.wandb_run_name,
                push_to_hub=False,
                max_grad_norm=self.config.max_grad_norm
            )
        
        if self.is_encoder_decoder:
            # Determine optimizer: Adafactor for mT5 (recommended), AdamW for others
            model_id_lower = self.config.model_id.lower()
            
            # Auto-select optimizer based on model
            if self.config.optimizer == "auto":
                use_adafactor = "mt5" in model_id_lower and not self.config.use_qlora
            elif self.config.optimizer == "adafactor":
                use_adafactor = not self.config.use_qlora
            else:
                use_adafactor = False
            
            if use_adafactor:
                optimizer = Adafactor(
                    self.model.parameters(),
                    scale_parameter=False,
                    relative_step=False,
                    warmup_init=False,
                    lr=self.config.learning_rate
                )
                print(f"  Using Adafactor optimizer (recommended for mT5)")
                optimizers = (optimizer, None)  # (optimizer, scheduler)
            else:
                optimizers = (None, None)  # Use default AdamW
            
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                optimizers=optimizers,  # Inject custom optimizer if needed
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
        
        print(f"\nStarting training...")
        print(f"   Epochs: {self.config.epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"   Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print()
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        # Getting the test results
        metrics = self.evaluate_test_set(dataset_dict['test'])
        
        # Saveing it to my central results csv -> I can access it later on
        self.save_results_to_csv(metrics, dataset_dict)
        wandb.finish()
        print("\n Training completed successfully!")
    
    def evaluate_test_set(self, test_dataset):
        """Evaluates the model on the test set with full metrics"""
        print("\n=== Test Set Evaluation ===")
        
        # Generate predictions
        print("Generating predictions...")
        self.model.eval()
        
        predictions = []
        references = test_dataset['summary_de']
        inputs = test_dataset['latin_text']
        charter_ids = test_dataset['nummer']
        
        with torch.no_grad():
            for i in tqdm(range(0, len(test_dataset), self.config.batch_size), desc="Predicting"):
                batch_inputs = inputs[i:i+self.config.batch_size]
                
                if self.is_encoder_decoder:
                    model_id_lower = self.config.model_id.lower()
                    
                    # I have to use for the final inference also the needed prefixes...
                    if "madlad" in model_id_lower:
                        task_prefix = "<2de> " 
                    elif "philta" in model_id_lower or "bowphs" in model_id_lower:
                        task_prefix = ""  # NO prefix for PhilTa
                    elif "mt5" in model_id_lower:
                        task_prefix = ""  # NO prefix for mT5
                    elif "t5" in model_id_lower:
                        task_prefix = "summarize: "
                    elif "bart" in model_id_lower:
                        task_prefix = ""
                    else:
                        task_prefix = ""
                    
                    if task_prefix:
                        batch_inputs = [task_prefix + text for text in batch_inputs]
                    
                    inputs_tokenized = self.tokenizer(
                        batch_inputs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_input_length
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        **inputs_tokenized,
                        max_length=self.config.max_target_length,
                        min_length=20,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    batch_predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                else:
                    # Decoder-only generation
                    # Use same format as training: Latin text + separator
                    prompts = [f"{text}\n\n" for text in batch_inputs]
                    
                    inputs_tokenized = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=False
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        **inputs_tokenized,
                        max_new_tokens=self.config.max_target_length,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # I am only interested in the generated german summary..
                    full_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    batch_predictions = []
                    for full_output, prompt in zip(full_outputs, prompts):
                        # Remove the prompt part
                        prediction = full_output[len(prompt):].strip()
                        batch_predictions.append(prediction)
                
                predictions.extend(batch_predictions)
        
        # In order for the judge-model to load, I need to free up memory, else OOM problems
        if self.config.use_expert_rules:
            print("\nFreeing GPU memory before expert rule evaluation...")
            print(f"   Model device: {self.model.device}")
            
            # Explicitly unload LoRA adapters if QLoRA was used
            if self.config.use_qlora and hasattr(self.model, 'unload'):
                print("   Unloading LoRA adapters...")
                self.model = self.model.unload()
            
            self.model.to('cpu')
            del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            
            # Force garbage collection multiple times
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            print(f"   Model and tokenizer deleted")
            print(f"   CUDA cache cleared")
            print(f"   Garbage collection run")
            
            # Check GPU memory
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - allocated
                print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
        
        # Evaluate with full metrics
        print("\nCalculating metrics...")
        evaluator = SummarizationEvaluator(
            bertscore_model="magistermilitum/bert_medieval_multilingual",
            bertscore_num_layers=12,
            mauve_model="magistermilitum/bert_medieval_multilingual",
            verbose=True
        )
        
        # Get tokenizer for tokenizer metrics (recreate a lightweight tokenizer for evaluation)
        from transformers import AutoTokenizer
        tokenizer_for_eval = AutoTokenizer.from_pretrained(self.config.model_id, trust_remote_code=True)
        
        metrics = evaluator.evaluate(
            predictions=predictions,
            references=references,
            inputs=inputs,
            use_expert_rules=self.config.use_expert_rules,
            tokenizer_id_or_obj=tokenizer_for_eval
        )
        
        wandb.log({
            "test/rouge1": metrics.rouge1,
            "test/rouge2": metrics.rouge2,
            "test/rougeL": metrics.rougeL,
            "test/bleu_score": metrics.bleu_score,
            "test/bertscore_precision_avg": metrics.bertscore_precision_avg,
            "test/bertscore_recall_avg": metrics.bertscore_recall_avg,
            "test/bertscore_f1_avg": metrics.bertscore_f1_avg,
            "test/cross_bertscore_f1_avg": metrics.cross_bertscore_f1_avg,
            "test/mauve_score": metrics.mauve_score,
            "test/cross_mauve_score": metrics.cross_mauve_score,
            "test/length_ratio": metrics.length_ratio,
            "test/avg_prediction_length": metrics.avg_prediction_length,
            "test/avg_reference_length": metrics.avg_reference_length
        })
        
        # Log tokenizer metrics if available
        if metrics.tokenizer_metrics:
            for key, value in metrics.tokenizer_metrics.items():
                wandb.log({f"test/tokenizer_metrics.{key}": value})
        
        # Log expert violations if available
        if metrics.expert_violations:
            for rule, rate in metrics.expert_violations.items():
                wandb.log({f"test/expert_violations.{rule}": rate})
        
        # Compute per-sample ROUGE and BLEU scores
        print("\nComputing per-sample metrics for detailed table...")
        from evaluate import load
        rouge_metric = load("rouge")
        bleu_metric = load("bleu")
        
        per_sample_rouge1 = []
        per_sample_rouge2 = []
        per_sample_rougeL = []
        per_sample_bleu = []
        per_sample_pred_length = []
        per_sample_ref_length = []
        per_sample_length_ratio = []
        
        for pred, ref in zip(predictions, references):
            # ROUGE per sample
            rouge_result = rouge_metric.compute(predictions=[pred], references=[ref])
            per_sample_rouge1.append(rouge_result['rouge1'])
            per_sample_rouge2.append(rouge_result['rouge2'])
            per_sample_rougeL.append(rouge_result['rougeL'])
            
            # BLEU per sample
            bleu_result = bleu_metric.compute(predictions=[pred], references=[[ref]])
            per_sample_bleu.append(bleu_result['bleu'])
            
            # Length metrics
            pred_len = len(pred.split())
            ref_len = len(ref.split())
            per_sample_pred_length.append(pred_len)
            per_sample_ref_length.append(ref_len)
            per_sample_length_ratio.append(pred_len / ref_len if ref_len > 0 else 0.0)
        
        metrics_table = wandb.Table(
            columns=[
                "charter_id", "sample_idx",
                "rouge1", "rouge2", "rougeL", "bleu",
                "bertscore_precision", "bertscore_recall", "bertscore_f1",
                "pred_length", "ref_length", "length_ratio",
                "input_preview", "prediction_preview", "reference_preview"
            ]
        )
        
        for i, (charter_id, inp, pred, ref) in enumerate(zip(charter_ids, inputs, predictions, references)):
            metrics_table.add_data(
                charter_id,
                i,
                round(per_sample_rouge1[i], 4),
                round(per_sample_rouge2[i], 4),
                round(per_sample_rougeL[i], 4),
                round(per_sample_bleu[i], 4),
                round(metrics.bertscore_precision_list[i], 4),
                round(metrics.bertscore_recall_list[i], 4),
                round(metrics.bertscore_f1_list[i], 4),
                per_sample_pred_length[i],
                per_sample_ref_length[i],
                round(per_sample_length_ratio[i], 4),
                inp[:200] + "..." if len(inp) > 200 else inp,
                pred[:200] + "..." if len(pred) > 200 else pred,
                ref[:200] + "..." if len(ref) > 200 else ref
            )
        
        wandb.log({"test/detailed_metrics_per_sample": metrics_table})
        
        table = wandb.Table(columns=["Charter ID", "Input (Latin)", "Prediction", "Reference"])
        for i in range(min(50, len(predictions))):  # Log first 50 examples
            table.add_data(
                charter_ids[i],
                inputs[i][:500],  # Truncate long Latin texts
                predictions[i],
                references[i]
            )
        wandb.log({"test/predictions": table})
        print("\nTest evaluation complete")
        return metrics
    
    def save_results_to_csv(self, metrics: EvaluationMetrics, dataset_dict: DatasetDict):
        # Determine more specific model type
        model_id_lower = self.config.model_id.lower()
        if self.is_encoder_decoder:
            if "philta" in model_id_lower or "bowphs" in model_id_lower:
                model_type = "encoder-decoder-philta"
            elif "madlad" in model_id_lower:
                model_type = "encoder-decoder-t5"
            elif "mt5" in model_id_lower:
                model_type = "encoder-decoder-mt5"
            elif "longt5" in model_id_lower:
                model_type = "encoder-decoder-longt5"
            else:
                model_type = "encoder-decoder"
        else:
            model_type = "decoder-only"
        
        # Prepare results record
        # Determine which optimizer was actually used
        if self.is_encoder_decoder:
            if self.config.optimizer == "auto":
                optimizer_used = "Adafactor" if "mt5" in model_id_lower and not self.config.use_qlora else "AdamW"
            elif self.config.optimizer == "adafactor" and not self.config.use_qlora:
                optimizer_used = "Adafactor"
            else:
                optimizer_used = "AdamW"
        else:
            optimizer_used = "AdamW"  # Decoder-only models use AdamW
        
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_name": self.config.wandb_run_name,
            "wandb_url": self.wandb_run_url,
            "model_id": self.config.model_id,
            "model_type": model_type,
            "num_parameters": self.num_model_parameters,
            "num_parameters_millions": round(self.num_model_parameters / 1e6, 2) if self.num_model_parameters else None,
            "dataset_path": self.config.dataset_path,
            "train_samples": len(dataset_dict['train']),
            "val_samples": len(dataset_dict['validation']),
            "test_samples": len(dataset_dict['test']),
            "debug_mode": self.config.debug_mode,
            "debug_ratio": self.config.debug_sample_ratio if self.config.debug_mode else None,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "effective_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "optimizer": optimizer_used,  # Track which optimizer was used
            "warmup_steps": self.config.warmup_steps,
            "weight_decay": self.config.weight_decay,
            "max_grad_norm": self.config.max_grad_norm,
            "label_smoothing": self.config.label_smoothing,
            "max_input_length": self.config.max_input_length,
            "max_target_length": self.config.max_target_length,
            "rouge1": round(metrics.rouge1, 4),
            "rouge2": round(metrics.rouge2, 4),
            "rougeL": round(metrics.rougeL, 4),
            "bleu": round(metrics.bleu_score, 4),
            "bertscore_precision": round(metrics.bertscore_precision_avg, 4),
            "bertscore_recall": round(metrics.bertscore_recall_avg, 4),
            "bertscore_f1": round(metrics.bertscore_f1_avg, 4),
            "mauve": round(metrics.mauve_score, 4) if metrics.mauve_score else None,
            "length_ratio": round(metrics.length_ratio, 4),
            "expert_rules_used": self.config.use_expert_rules,
        }
        
        if metrics.expert_violations:
            for rule, rate in metrics.expert_violations.items():
                results[f"expert_{rule}"] = round(rate, 4)
        
        results["output_dir"] = self.config.output_dir
        
        if self.results_csv_path.exists():
            df_existing = pd.read_csv(self.results_csv_path)
            df_new = pd.DataFrame([results])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = pd.DataFrame([results])
        
        df_combined.to_csv(self.results_csv_path, index=False)
        print(f"   ✓ Results appended to {self.results_csv_path}")
        print(f"   📊 Total runs: {len(df_combined)}")
    
    def evaluate_only(self):
        """Only evaluate an existing model (no training)"""
        print("Evaluation-only mode")
        
        wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=f"{self.config.wandb_run_name}-eval",
            config=self.config.__dict__
        )
        self.wandb_run_url = wandb_run.get_url()
        
        dataset_dict = self.load_and_split_data()
        
        print(f"\nLoading model: {self.config.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_id).to(self.device)
            self.is_encoder_decoder = True
            print("   Loaded as Encoder-Decoder model")
        except:
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id).to(self.device)
            self.is_encoder_decoder = False
            print("   Loaded as Decoder-only model")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"   Total parameters: {self.num_model_parameters:,}")
        
        metrics = self.evaluate_test_set(dataset_dict['test'])
        
        self.save_results_to_csv(metrics, dataset_dict)
        
        wandb.finish()
        print("\nEvaluation completed!")
        print(f"Results saved to: {self.results_csv_path}")


def build_slurm_python_args(config: FinetuningConfig) -> List[str]:
    """Builds Python arguments for SLURM submission"""
    args = [
        "--model_id", config.model_id,
        "--dataset", config.dataset_path,
        "--output_dir", config.output_dir,
        "--epochs", str(config.epochs),
        "--batch_size", str(config.batch_size),
        "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
        "--learning_rate", str(config.learning_rate),
        "--run_name", config.wandb_run_name,
        "--run_mode", "local"
    ]
    
    if config.eval_only:
        args.append("--eval_only")
    
    if config.debug_mode:
        args.extend(["--debug", "--debug_ratio", str(config.debug_sample_ratio)])
    
    if config.use_expert_rules:
        args.append("--use_expert_rules")
    
    if config.use_qlora:
        args.extend([
            "--use_qlora",
            "--lora_r", str(config.lora_r),
            "--lora_alpha", str(config.lora_alpha),
            "--lora_dropout", str(config.lora_dropout)
        ])
    
    return args


def main():
    parser = argparse.ArgumentParser(description="End-to-End Latin→German Summarization Finetuning")

    parser.add_argument("--model_id", type=str, default="bowphs/PhilTa",
                        help="Base model to finetune")
    parser.add_argument("--dataset", type=str, 
                        default="../data/all.csv",
                        help="Path to training CSV")
    parser.add_argument("--output_dir", type=str, 
                        default="./models/philta-finetuned",
                        help="Output directory for finetuned model")
    

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--max_target_length", type=int, default=256)
    
    parser.add_argument("--eval_only", action="store_true", 
                        help="Only evaluate, don't train")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--use_expert_rules", action="store_true",
                        help="Use expert rules evaluation (slow, requires Qwen3-8B)")
    
    # Optimizer options
    parser.add_argument("--optimizer", type=str, default="auto",
                        choices=["auto", "adamw", "adafactor"],
                        help="Optimizer to use. 'auto' selects Adafactor for mT5, AdamW for others")
    
    # QLoRA options
    parser.add_argument("--use_qlora", action="store_true",
                        help="Use QLoRA (4-bit quantization with LoRA adapters)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")
    
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: Use only 10% of samples for faster testing")
    parser.add_argument("--debug_ratio", type=float, default=0.1,
                        help="Ratio of samples to use in debug mode (default: 0.1)")
    
    parser.add_argument("--run_name", type=str, default="end2end-latin-german")
    
    # SLURM args
    parser.add_argument("--run_mode", type=str, choices=["local", "slurm"], default="local")
    parser.add_argument("--slurm_partition", type=str, default="mcml-dgx-a100-40x8")
    parser.add_argument("--slurm_gpus", type=int, default=1)
    parser.add_argument("--slurm_time", type=str, default="06:00:00")
    parser.add_argument("--slurm_job_name", type=str, default=None)
    parser.add_argument("--slurm_log_dir", type=str, default="logs")
    parser.add_argument("--slurm_conda_env", type=str, default="llm-thesis")
    parser.add_argument("--slurm_email", type=str, default=None)
    
    args = parser.parse_args()
    
    config = FinetuningConfig(
        model_id=args.model_id,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        label_smoothing=args.label_smoothing,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        eval_only=args.eval_only,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        use_expert_rules=args.use_expert_rules,
        optimizer=args.optimizer,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        debug_mode=args.debug,
        debug_sample_ratio=args.debug_ratio,
        wandb_run_name=args.run_name,
        run_mode=args.run_mode,
        slurm_partition=args.slurm_partition,
        slurm_gpus=args.slurm_gpus,
        slurm_time=args.slurm_time,
        slurm_job_name=args.slurm_job_name
    )
    
    if args.run_mode == "slurm":
        # Submit to SLURM
        job_name = args.slurm_job_name or f"e2e-finetune-{args.run_name}"
        
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
        print(f"Model: {config.model_id}")
        print(f"Dataset: {config.dataset_path}")
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
        finetuner = End2EndFinetuner(config)
        
        if config.eval_only:
            finetuner.evaluate_only()
        else:
            finetuner.train()


if __name__ == "__main__":
    main()
