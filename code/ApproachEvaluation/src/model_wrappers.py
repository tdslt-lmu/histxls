#!/usr/bin/env python3
"""
Model Wrappers for Different LLM Architectures
"""

import os
import sys
import time
import torch
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    BitsAndBytesConfig
)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class BaseModelWrapper(ABC):
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_prompt, self.blacklisted_charter_ids, self.whitelisted_charter_ids = self._load_system_prompt()
    
    def _load_system_prompt(self) -> Tuple[str, List[str], List[str]]:
        # Check if config has system_prompts_file attribute
        if not hasattr(self.config, 'system_prompts_file'):
            return self._get_fallback_prompt(), [], []
        
        prompts_file = Path(self.config.system_prompts_file)
        
        # If relative path: relative to script directory
        if not prompts_file.is_absolute():
            script_dir = Path.cwd()  # Fallback to current directory
            prompts_file = script_dir / prompts_file
        
        if not prompts_file.exists():
            return self._get_fallback_prompt(), [], []
        
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
            
            prompts = prompts_data.get("prompts", [])
            
            if not prompts:
                raise ValueError("No prompts found in JSON file")
            
            # Sort prompts by version (newest first)
            prompts_sorted = sorted(
                prompts,
                key=lambda p: self._parse_version(p.get("version", "0.0")),
                reverse=True
            )
            
            # Select prompt based on version
            system_prompt_version = getattr(self.config, 'system_prompt_version', None)
            
            if system_prompt_version:
                # Search for specific version
                selected_prompt = next(
                    (p for p in prompts if p.get("version") == system_prompt_version),
                    None
                )
                if selected_prompt is None:
                    available_versions = [p.get("version") for p in prompts]
                    raise ValueError(
                        f"System Prompt Version '{system_prompt_version}' not found. "
                        f"Available versions: {available_versions}"
                    )
            else:
                # Use newest version
                selected_prompt = prompts_sorted[0]
            
            version = selected_prompt.get("version", "unknown")
            description = selected_prompt.get("description", "")
            prompt_text = selected_prompt.get("system_prompt", "")
            blacklisted_ids = selected_prompt.get("blacklisted_charter_ids", [])
            whitelisted_ids = selected_prompt.get("whitelisted_charter_ids", [])
            
            print(f"  System Prompt loaded: Version {version}")
            print(f"  Description: {description}")
            
            if whitelisted_ids:
                print(f"  Whitelisted Charter IDs: {len(whitelisted_ids)} ({', '.join(whitelisted_ids[:5])}{'...' if len(whitelisted_ids) > 5 else ''})")
                print(f"  Whitelist active: Only these charters will be evaluated (blacklist ignored)")
            elif blacklisted_ids:
                print(f"  Blacklisted Charter IDs: {len(blacklisted_ids)} ({', '.join(blacklisted_ids[:5])}{'...' if len(blacklisted_ids) > 5 else ''})")
            
            return prompt_text, blacklisted_ids, whitelisted_ids
            
        except Exception as e:
            print(f"  Error loading system prompts: {e}")
    
    @staticmethod
    def _get_fallback_prompt() -> str:
        return "Du bist ein Experte für mittelalterliches Latein. Deine Aufgabe ist es, lateinische Urkunden prägnant auf Deutsch zusammenzufassen. Antworte NUR mit der deutschen Zusammenfassung."
    
    @staticmethod
    def _parse_version(version_str: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x) for x in version_str.split('.'))
        except:
            return (0,)
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def create_prompt(self, sample: Dict, train_dataset: Any, n_shots: int) -> Tuple[str, str]:
        pass
    
    @abstractmethod
    def generate(self, prompt: str, generation_config: GenerationConfig) -> str:
        pass
    
    def prepare_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None,
            eos_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
        )


class CausalLMWrapper(BaseModelWrapper):
    """Wrapper for decoder-only models"""
    
    def load_model(self):
        print(f"Loading Causal LM: {self.config.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        
        if self.config.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=(self.config.quantization_bits == 4),
                load_in_8bit=(self.config.quantization_bits == 8),
                bnb_4bit_quant_type="nf4" if self.config.quantization_bits == 4 else None,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            
            num_gpus = torch.cuda.device_count()
            max_memory = {i: self.config.max_memory_per_gpu for i in range(num_gpus)}
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                quantization_config=quantization_config,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="offload",
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        num_gpus = torch.cuda.device_count()
        print(f"Model loaded on {num_gpus} GPU(s)")
    
    def create_prompt(self, sample: Dict, train_dataset: Any, n_shots: int) -> Tuple[str, str]:
        if self.config.use_chat_template:
            return self._create_chat_prompt(sample, train_dataset, n_shots)
        else:
            return self._create_simple_prompt(sample, train_dataset, n_shots)
    
    def _create_chat_prompt(self, sample: Dict, train_dataset: Any, n_shots: int) -> Tuple[str, str]:
        system_message = {
            "role": "system",
            "content": self.system_prompt
        }
        
        messages = [system_message]
        
        # Add few-shot examples
        examples = train_dataset.filter(
            lambda ex: ex['nummer'] != sample['nummer']
        ).shuffle(seed=42).select(range(n_shots))
        
        for ex in examples:
            messages.append({
                "role": "user",
                "content": f"Lateinischer Text:\n{ex['latin_text']}"
            })
            messages.append({
                "role": "assistant",
                "content": f"Deutsche Zusammenfassung:\n{ex['summary_de']}"
            })
        
        # Add test sample
        messages.append({
            "role": "user",
            "content": f"Lateinischer Text:\n{sample['latin_text']}"
        })
        
        # Guidance: Start response
        messages.append({
            "role": "assistant",
            "content": "Deutsche Zusammenfassung:\n"
        })
        
        prompt_string = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return prompt_string, sample['summary_de']
    
    def _create_simple_prompt(self, sample: Dict, train_dataset: Any, n_shots: int) -> Tuple[str, str]:
        examples = train_dataset.filter(
            lambda ex: ex['nummer'] != sample['nummer']
        ).shuffle(seed=42).select(range(n_shots))
        
        prompt = "Aufgabe: Fasse den folgenden mittelalterlichen lateinischen Text als einzelne, prägnante deutsche Zusammenfassung zusammen.\n\n"
        
        for ex in examples:
            prompt += f"--- Beispiel ---\n"
            prompt += f"Lateinischer Text:\n{ex['latin_text']}\n\n"
            prompt += f"Deutsche Zusammenfassung:\n{ex['summary_de']}\n\n"
        
        prompt += "--- Aufgabe ---\n"
        prompt += f"Lateinischer Text:\n{sample['latin_text']}\n\n"
        prompt += f"Deutsche Zusammenfassung:\n"
        
        return prompt, sample['summary_de']
    
    def generate(self, prompt: str, generation_config: GenerationConfig) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length
        )
        input_token_len = inputs.input_ids.shape[1]
        
        # Move inputs to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation
        output_ids = self.model.generate(**inputs, generation_config=generation_config)
        
        # Decoding (only new tokens)
        prediction = self.tokenizer.decode(
            output_ids[0][input_token_len:],
            skip_special_tokens=True
        )
        
        # Post-processing
        # Remove <think>...</think> blocks (e.g. for Qwen models)
        import re
        prediction = re.sub(r'<think>.*?</think>', '', prediction, flags=re.DOTALL).strip()
        
        prediction = prediction.strip()
        prediction = prediction.removesuffix("}").strip()
        prediction = prediction.removesuffix('"').strip()
        if prediction.startswith("assistant\n\n"):
            prediction = prediction[len("assistant\n\n"):].strip()
        
        return prediction


class Seq2SeqWrapper(BaseModelWrapper):
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_id)
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        else:
    
    def create_prompt(self, sample: Dict, train_dataset: Any, n_shots: int) -> Tuple[str, str]:
        examples = train_dataset.filter(
            lambda ex: ex['nummer'] != sample['nummer']
        ).shuffle(seed=42).select(range(n_shots))
        
        prompt = "Aufgabe: Fasse den folgenden mittelalterlichen lateinischen Text als einzelne, prägnante deutsche Zusammenfassung zusammen.\n\n"
        
        for ex in examples:
            prompt += f"--- Beispiel ---\n"
            prompt += f"Lateinischer Text:\n{ex['latin_text']}\n\n"
            prompt += f"Deutsche Zusammenfassung:\n{ex['summary_de']}\n\n"
        
        prompt += "--- Aufgabe ---\n"
        prompt += f"Lateinischer Text:\n{sample['latin_text']}\n\n"
        prompt += f"Deutsche Zusammenfassung:\n"
        
        return prompt, sample['summary_de']
    
    def generate(self, prompt: str, generation_config: GenerationConfig) -> str:
        """Generates text with seq2seq model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_length
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        output_ids = self.model.generate(**inputs, generation_config=generation_config)
        prediction = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return prediction.strip()
    
    def prepare_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )


class APIModelWrapper(BaseModelWrapper):
    """Wrapper for API-based models (Gemini). Actually not used in the final thesis..."""
    
    def __init__(self, config):
        super().__init__(config)
        self.api_client = None
        self.last_request_time = 0
        self.min_request_interval = 60.0 / config.api_requests_per_minute
    
    def load_model(self):
        """Initializes API connection"""
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI library not installed. "
                "Install it with: pip install google-generativeai"
            )
        
        print(f"Initializing API Model: {self.config.model_id}")
        
        # Load API key from environment variable
        api_key_var = self.config.api_key_env_var or "GEMINI_API_KEY"
        api_key = os.environ.get(api_key_var)
        
        if not api_key:
            raise ValueError(
                f"API Key not found! Please set environment variable '{api_key_var}'.\n"
                f"Example: export {api_key_var}='your-api-key-here'"
            )
        
        # Gemini API
        if "gemini" in self.config.model_id.lower():
            genai.configure(api_key=api_key)
            
            # Initialize model with system_instruction (loaded prompt)
            self.api_client = genai.GenerativeModel(
                self.config.model_id,
                system_instruction=self.system_prompt
            )
            print(f"✓ Gemini API initialized: {self.config.model_id}")
            print(f"  System Instruction active")
        
        # Additional APIs can be added here (OpenAI, Anthropic, etc.)
        else:
            raise ValueError(f"API model '{self.config.model_id}' not yet supported")
        
        # No tokenizer needed for APIs, but set device for compatibility
        self.device = "api"
    
    def create_prompt(self, sample: Dict, train_dataset: Any, n_shots: int) -> Tuple[str, str]:
        # Collect few-shot examples
        examples = train_dataset.filter(
            lambda ex: ex['nummer'] != sample['nummer']
        ).shuffle(seed=42).select(range(n_shots))
        
        # Prompt without system instruction (already set in model init)
        prompt_parts = []
        
        if n_shots > 0:
            prompt_parts.append("Hier sind Beispiele für meine Aufgabe:\n")
            
            for i, ex in enumerate(examples, 1):
                prompt_parts.append(f"\nBeispiel {i}:")
                prompt_parts.append(f"Latein: {ex['latin_text']}")
                prompt_parts.append(f"Deutsch: {ex['summary_de']}\n")
        
        prompt_parts.append(f"\nFasse nun diesen lateinischen Text auf Deutsch zusammen:")
        prompt_parts.append(sample['latin_text'])
        
        prompt = "\n".join(prompt_parts)
        
        return prompt, sample['summary_de']
    
    def _rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def generate(self, prompt: str, generation_config: GenerationConfig = None) -> str:
        self._rate_limit()
        
        try:
            if "gemini" in self.config.model_id.lower():
                # Gemini API call with relaxed safety settings
                from google.generativeai.types import HarmCategory, HarmBlockThreshold
                
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                
                generation_config_dict = {
                    "temperature": self.config.api_temperature,
                    "top_p": self.config.api_top_p,
                    "max_output_tokens": self.config.max_new_tokens,
                }
                
                response = self.api_client.generate_content(
                    prompt,
                    generation_config=generation_config_dict,
                    safety_settings=safety_settings
                )
                
                prediction = None
                try:
                    if hasattr(response, 'text') and response.text:
                        prediction = response.text.strip()
                except ValueError as e:
                    # response.text throws ValueError if blocked
                    print(f"⚠️  response.text not available: {e}")
                
                # If that doesn't work, manually extract from candidates
                if prediction is None and response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    
                    # Try to extract text from parts
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        parts_text = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                parts_text.append(part.text)
                        
                        if parts_text:
                            prediction = "".join(parts_text).strip()
                
                # If still no text, return error
                if not prediction:
                    if response.candidates and len(response.candidates) > 0:
                        candidate = response.candidates[0]
                        if candidate.finish_reason == 2:
                            return "[BLOCKED BY SAFETY FILTER]"
                        elif candidate.finish_reason == 3:
                            return "[BLOCKED BY RECITATION]"
                        else:
                            return f"[NO CONTENT: finish_reason={candidate.finish_reason}]"
                    else:
                        return "[NO CANDIDATES IN RESPONSE]"
                
            else:
                raise ValueError(f"API model '{self.config.model_id}' not yet supported")
            
            return prediction
            
        except Exception as e:
            print(f"Error in API request: {e}")
            import traceback
            traceback.print_exc()
            return "[API ERROR]"
    
    def prepare_generation_config(self) -> GenerationConfig:
        return None
