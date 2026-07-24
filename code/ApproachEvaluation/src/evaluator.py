#!/usr/bin/env python3
"""
Central Evaluation Module for Text Summarization
"""

import evaluate
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, asdict
import json
import re
import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    rouge1: float
    rouge2: float
    rougeL: float
    bleu_score: float
    bertscore_precision_avg: float
    bertscore_recall_avg: float
    bertscore_f1_avg: float
    mauve_score: float
    avg_prediction_length: float
    avg_reference_length: float
    length_ratio: float
    cross_bertscore_f1_avg: Optional[float] = None
    cross_mauve_score: Optional[float] = None
    expert_violations: Optional[Dict[str, float]] = None  # Average violation rate
    tokenizer_metrics: Optional[Dict[str, float]] = None  # Tokenizer analysis (Input Text)
    
    # Per-sample score lists for distributions (optional)
    rouge1_list: Optional[List[float]] = None
    rouge2_list: Optional[List[float]] = None
    rougeL_list: Optional[List[float]] = None
    bertscore_precision_list: Optional[List[float]] = None
    bertscore_recall_list: Optional[List[float]] = None
    bertscore_f1_list: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def __str__(self) -> str:
        lines = [
            "Evaluation Metrics:",
            f"  ROUGE-1: {self.rouge1:.4f}",
            f"  ROUGE-2: {self.rouge2:.4f}",
            f"  ROUGE-L: {self.rougeL:.4f}",
            f"  BLEU:    {self.bleu_score:.2f}",
            f"  BERTScore:",
            f"    - Precision: {self.bertscore_precision_avg:.4f}",
            f"    - Recall:    {self.bertscore_recall_avg:.4f}",
            f"    - F1:        {self.bertscore_f1_avg:.4f}",
            f"  MAUVE:   {self.mauve_score:.4f}",
            f"  Length (Chars):",
            f"    - Pred Ø:    {self.avg_prediction_length:.1f}",
            f"    - Ref Ø:     {self.avg_reference_length:.1f}",
            f"    - Ratio:     {self.length_ratio:.2f}",
            f"  Cross-Lingual (Input vs. Prediction):",
            f"    - BERTScore-F1: {self.cross_bertscore_f1_avg:.4f}" if self.cross_bertscore_f1_avg is not None else "",
            f"    - MAUVE:        {self.cross_mauve_score:.4f}" if self.cross_mauve_score is not None else ""
        ]
        
        if self.tokenizer_metrics:
            lines.append("  Tokenizer Analysis (Input Text):")
            for k, v in self.tokenizer_metrics.items():
                lines.append(f"    - {k}: {v:.4f}")

        if self.expert_violations:
            lines.append("  Expert Rule Violations (Rate):")
            for k, v in self.expert_violations.items():
                lines.append(f"    - {k}: {v:.2%}")
        
        # Filter empty lines
        lines = [line for line in lines if line]
        return "\n".join(lines)


class SummarizationEvaluator:
    """
    Evaluator for text summarizations
    """
    
    def __init__(
        self,
        bertscore_model: str = "magistermilitum/bert_medieval_multilingual",
        bertscore_num_layers: int = 12,
        mauve_model: str = "magistermilitum/bert_medieval_multilingual",
        judge_model_id: str = "Qwen/Qwen3-8B",
        use_rouge: bool = True,
        use_bleu: bool = True,
        use_bertscore: bool = True,
        use_mauve: bool = True,
        verbose: bool = True
    ):
        self.bertscore_model = bertscore_model
        self.bertscore_num_layers = bertscore_num_layers
        self.mauve_model = mauve_model
        self.judge_model_id = judge_model_id
        self.use_rouge = use_rouge
        self.use_bleu = use_bleu
        self.use_bertscore = use_bertscore
        self.use_mauve = use_mauve
        self.verbose = verbose
        
        # Load metrics
        self._load_metrics()
    
    def _load_metrics(self):
        """Loads the required metrics"""
        if self.verbose:
            print("Loading evaluation metrics...")
        
        self.rouge_metric = None
        self.bleu_metric = None
        self.bertscore_metric = None
        self.mauve_metric = None
        
        if self.use_rouge:
            self.rouge_metric = evaluate.load('rouge')
            if self.verbose:
                print("  ROUGE loaded")
        
        if self.use_bleu:
            self.bleu_metric = evaluate.load('sacrebleu')
            if self.verbose:
                print("  SacreBLEU loaded")
        
        if self.use_bertscore:
            self.bertscore_metric = evaluate.load('bertscore')
            if self.verbose:
                print(f"  BERTScore loaded (model: {self.bertscore_model})")
        
        if self.use_mauve:
            self.mauve_metric = evaluate.load('mauve')
            if self.verbose:
                print(f"  MAUVE loaded (model: {self.mauve_model})")
        
        if self.verbose:
            print()
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        inputs: Optional[List[str]] = None,
        llm_predict_fn: Optional[Callable[[str], str]] = None,
        use_expert_rules: bool = False,
        tokenizer_id_or_obj: Optional[Any] = None
    ) -> EvaluationMetrics:
        if len(predictions) != len(references):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) != Number of references ({len(references)})"
            )
        
        if inputs and len(inputs) != len(predictions):
            raise ValueError(
                f"Number of inputs ({len(inputs)}) != Number of predictions ({len(predictions)})"
            )
        
        # Sanitize inputs: Replace None with empty string to prevent metric failures
        predictions = [p if p is not None else "" for p in predictions]
        references = [r if r is not None else "" for r in references]
        if inputs:
            inputs = [i if i is not None else "" for i in inputs]
        
        if self.verbose:
            print(f"Calculating metrics for {len(predictions)} samples...")
            print()
        
        # Calculate individual metrics
        rouge_scores = self._compute_rouge(predictions, references)
        bleu_score = self._compute_bleu(predictions, references)
        bertscore_scores = self._compute_bertscore(predictions, references)
        mauve_score = self._compute_mauve(predictions, references)
        length_metrics = self._compute_length_metrics(predictions, references)
        
        # Cross-Lingual Metrics (Input vs. Prediction)
        cross_bertscore_f1 = None
        cross_mauve = None
        tokenizer_metrics = None
        
        if inputs:
            if self.verbose:
                print("  Calculating cross-lingual metrics (Input vs. Prediction)...")
            
            # BERTScore (Input vs. Prediction)
            cross_bert_scores = self._compute_bertscore(predictions, inputs)
            cross_bertscore_f1 = cross_bert_scores['f1_avg']
            
            # MAUVE (Input vs. Prediction)
            cross_mauve = self._compute_mauve(predictions, inputs)

            # NEW: Tokenizer Analysis
            if tokenizer_id_or_obj:
                if self.verbose:
                    print("  Analyzing tokenizer efficiency on input texts...")
                tok_eval = TokenizerEvaluator(tokenizer_id_or_obj)
                tokenizer_metrics = tok_eval.evaluate(inputs)
                if self.verbose and tokenizer_metrics:
                    print(f"     Ø Chars/Token: {tokenizer_metrics['avg_chars_per_token']:.2f}")
                    print(f"     Single-Char Rate: {tokenizer_metrics['single_char_token_rate']:.2%}")
            
        # Expert Rule Evaluation (LLM-based)
        expert_violations = None
        
        # Case 1: External function provided
        if llm_predict_fn:
            if self.verbose:
                print("  Calculating expert rule violations (with external function)...")
            expert_evaluator = ExpertRuleEvaluator(llm_predict_fn, verbose=self.verbose)
            expert_violations = expert_evaluator.evaluate_batch(predictions)
            
        # Case 2: Request internal function
        elif use_expert_rules:
            print(f"DEBUG: Starting expert rule evaluation...")
            print(f"DEBUG: use_expert_rules={use_expert_rules}")
            print(f"DEBUG: Number of predictions to evaluate: {len(predictions)}")
            
            # CRITICAL: Aggressively clear GPU memory before loading judge model
            # BERTScore and MAUVE may have left models in GPU memory
            print(f"DEBUG: Performing final GPU cleanup before judge model loading...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()
            print(f"DEBUG: Final cleanup complete. GPU memory cleared.")
            
            if self.verbose:
                print(f"  Calculating expert rule violations (loading Judge: {self.judge_model_id})...")
            
            # Load Judge Model internally
            print(f"DEBUG: Loading judge model: {self.judge_model_id}...")
            judge_model, judge_tokenizer = self._load_judge_model()
            print(f"DEBUG: Judge model loaded successfully!")
            print(f"DEBUG: Judge model is not None: {judge_model is not None}")
            print(f"DEBUG: Judge tokenizer is not None: {judge_tokenizer is not None}")
            
            if judge_model and judge_tokenizer:
                print(f"DEBUG: Creating internal predict function...")
                # Create local predict function
                def _internal_predict_fn(prompt: str) -> str:
                    print(f"DEBUG: _internal_predict_fn called with prompt length: {len(prompt)}")
                    try:
                        gen_config = GenerationConfig(
                            max_new_tokens=1024,  # More tokens for <think>...</think> + answer
                            do_sample=False,
                            pad_token_id=judge_tokenizer.eos_token_id
                        )
                        
                        final_prompt = prompt
                        # Chat Template
                        if hasattr(judge_tokenizer, "apply_chat_template"):
                            try:
                                messages = [{"role": "user", "content": prompt}]
                                final_prompt = judge_tokenizer.apply_chat_template(
                                    messages, tokenize=False, add_generation_prompt=True
                                )
                            except Exception:
                                pass
                        
                        inputs = judge_tokenizer(final_prompt, return_tensors="pt").to(judge_model.device)
                        outputs = judge_model.generate(**inputs, generation_config=gen_config)
                        
                        # Decode only new tokens
                        input_len = inputs.input_ids.shape[1]
                        prediction = judge_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                        
                        # Qwen3 Thinking Mode: Remove <think>...</think> Tags
                        prediction = re.sub(r'<think>.*?</think>', '', prediction, flags=re.DOTALL)
                        
                        return prediction.strip()
                        
                    except Exception as e:
                        print(f"Error during judge inference: {e}")
                        return ""

                # Evaluate
                print(f"DEBUG: Creating ExpertRuleEvaluator...")
                expert_evaluator = ExpertRuleEvaluator(_internal_predict_fn, verbose=self.verbose)
                print(f"DEBUG: Starting evaluate_batch with {len(predictions)} predictions...")
                expert_violations = expert_evaluator.evaluate_batch(predictions)
                print(f"DEBUG: evaluate_batch completed. Results: {expert_violations}")
                
                # Unload Judge Model
                print(f"DEBUG: Unloading judge model...")
                self._unload_judge_model(judge_model, judge_tokenizer)
                print(f"DEBUG: Judge model unloaded successfully")
            else:
                print("Could not load judge model. Skipping expert evaluation.")

        if expert_violations and self.verbose:
            print("     Violations (Rates):")
            for k, v in expert_violations.items():
                print(f"       - {k}: {v:.2%}")
        
        # Create result container
        metrics = EvaluationMetrics(
            rouge1=rouge_scores['rouge1'],
            rouge2=rouge_scores['rouge2'],
            rougeL=rouge_scores['rougeL'],
            bleu_score=bleu_score,
            bertscore_precision_avg=bertscore_scores['precision_avg'],
            bertscore_recall_avg=bertscore_scores['recall_avg'],
            bertscore_f1_avg=bertscore_scores['f1_avg'],
            mauve_score=mauve_score,
            avg_prediction_length=length_metrics['avg_prediction_length'],
            avg_reference_length=length_metrics['avg_reference_length'],
            length_ratio=length_metrics['length_ratio'],
            cross_bertscore_f1_avg=cross_bertscore_f1,
            cross_mauve_score=cross_mauve,
            expert_violations=expert_violations,
            tokenizer_metrics=tokenizer_metrics,
            rouge1_list=rouge_scores.get('rouge1_list'),
            rouge2_list=rouge_scores.get('rouge2_list'),
            rougeL_list=rouge_scores.get('rougeL_list'),
            bertscore_precision_list=bertscore_scores.get('precision_list'),
            bertscore_recall_list=bertscore_scores.get('recall_list'),
            bertscore_f1_list=bertscore_scores.get('f1_list')
        )
        
        if self.verbose:
            print()
            print(metrics)
            print()
        
        return metrics
    
    def _compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        if not self.use_rouge or self.rouge_metric is None:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        if self.verbose:
            print("  Calculating ROUGE...")
        
        # Aggregate scores
        rouge_scores = self.rouge_metric.compute(
            predictions=predictions,
            references=references
        )
        
        # Per-sample scores for distributions
        rouge1_list = []
        rouge2_list = []
        rougeL_list = []
        
        for pred, ref in zip(predictions, references):
            sample_scores = self.rouge_metric.compute(
                predictions=[pred],
                references=[ref]
            )
            rouge1_list.append(sample_scores['rouge1'])
            rouge2_list.append(sample_scores['rouge2'])
            rougeL_list.append(sample_scores['rougeL'])
        
        # Add lists to return dict
        rouge_scores['rouge1_list'] = rouge1_list
        rouge_scores['rouge2_list'] = rouge2_list
        rouge_scores['rougeL_list'] = rougeL_list
        
        if self.verbose:
            print(f"     ROUGE-1: {rouge_scores['rouge1']:.4f}")
            print(f"     ROUGE-2: {rouge_scores['rouge2']:.4f}")
            print(f"     ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        return rouge_scores
    
    def _compute_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        if not self.use_bleu or self.bleu_metric is None:
            return 0.0
        
        if self.verbose:
            print("  Calculating BLEU...")
        
        # SacreBLEU expects references as list of lists
        references_formatted = [[ref] for ref in references]
        
        bleu_scores = self.bleu_metric.compute(
            predictions=predictions,
            references=references_formatted
        )
        
        bleu_score = bleu_scores['score']
        
        if self.verbose:
            print(f"     BLEU: {bleu_score:.2f}")
        
        return bleu_score
    
    def _compute_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        if not self.use_bertscore or self.bertscore_metric is None:
            return {
                'precision_avg': 0.0,
                'recall_avg': 0.0,
                'f1_avg': 0.0,
                'precision_list': [],
                'recall_list': [],
                'f1_list': []
            }
        
        if self.verbose:
            print(f"  Calculating BERTScore (model: {self.bertscore_model})...")
        
        try:
            bert_scores = self.bertscore_metric.compute(
                predictions=predictions,
                references=references,
                model_type=self.bertscore_model,
                num_layers=self.bertscore_num_layers
            )
            
            # Average over all samples
            precision_avg = sum(bert_scores['precision']) / len(bert_scores['precision'])
            recall_avg = sum(bert_scores['recall']) / len(bert_scores['recall'])
            f1_avg = sum(bert_scores['f1']) / len(bert_scores['f1'])
            
            # Convert tensors to Python lists for proper serialization
            precision_list = [float(x) for x in bert_scores['precision']]
            recall_list = [float(x) for x in bert_scores['recall']]
            f1_list = [float(x) for x in bert_scores['f1']]
            
            if self.verbose:
                print(f"     Precision (avg): {precision_avg:.4f}")
                print(f"     Recall (avg):    {recall_avg:.4f}")
                print(f"     F1 (avg):        {f1_avg:.4f}")
            
            return {
                'precision_avg': precision_avg,
                'recall_avg': recall_avg,
                'f1_avg': f1_avg,
                'precision_list': precision_list,
                'recall_list': recall_list,
                'f1_list': f1_list
            }
            
        except Exception as e:
            print(f"  BERTScore calculation failed: {e}")
            
            return {
                'precision_avg': 0.0,
                'recall_avg': 0.0,
                'f1_avg': 0.0,
                'precision_list': [],
                'recall_list': [],
                'f1_list': []
            }
    
    def _compute_mauve(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Calculates MAUVE metric with text normalization (Upper/lowercase, punctuation, whitespace)
        """
        if not self.use_mauve or self.mauve_metric is None:
            return 0.0
        
        if self.verbose:
            print("  Calculating MAUVE (with text normalization)...")
        
        try:
            # Text normalization for more robust MAUVE scores
            predictions_normalized = [self._normalize_text_for_mauve(p) for p in predictions]
            references_normalized = [self._normalize_text_for_mauve(r) for r in references]
            
            if self.verbose:
                print(f"     Normalization: {len(predictions)} texts processed")
                print(f"     Model: {self.mauve_model}")
                
                # Show some example pairs (max 3)
                num_examples = min(3, len(predictions))
                print(f"     Example pairs (normalized, first {num_examples}):")
                for i in range(num_examples):
                    print(f"       [{i+1}] Prediction: {predictions_normalized[i]}")
                    print(f"           Reference:  {references_normalized[i]}")
            
            # MAUVE-spezifische Parameter anpassen
            # CRITICAL: FAISS requires num_buckets <= num_samples
            # Otherwise: "Number of training points (N) should be at least as large as number of clusters (K)"
            num_samples = len(predictions)
            
            # Use theoretical recommendation from MAUVE paper:
            # Optimal quantization size k should scale proportionally to the cube root of n
            # k = Θ(n^(1/3))
            num_buckets = max(2, int(num_samples ** (1/3)))
            
            # Ensure num_buckets doesn't exceed num_samples (FAISS constraint)
            num_buckets = min(num_buckets, num_samples)
            
            if self.verbose:
                print(f"     MAUVE Config: num_buckets={num_buckets} (from n^(1/3) formula), samples={num_samples}")
            
            mauve_results = self.mauve_metric.compute(
                predictions=predictions_normalized,
                references=references_normalized,
                featurize_model_name=self.mauve_model,
                max_text_length=512,  # Maximum text length
                num_buckets=num_buckets,  # Adaptive number of buckets
                device_id=0 if self._has_gpu() else -1,  # GPU if available
                verbose=False  # Internal MAUVE logging off
            )
            
            mauve_score = mauve_results.mauve
            
            return mauve_score
        
        except Exception as e:
            if self.verbose:
                print(f"     MAUVE calculation failed: {e}")
            return 0.0
    
    def _compute_length_metrics(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        if not predictions:
            return {
                'avg_prediction_length': 0.0,
                'avg_reference_length': 0.0,
                'length_ratio': 0.0
            }
            
        pred_lens = [len(p) for p in predictions]
        ref_lens = [len(r) for r in references]
        
        avg_pred = sum(pred_lens) / len(pred_lens)
        avg_ref = sum(ref_lens) / len(ref_lens)
        
        ratio = avg_pred / avg_ref if avg_ref > 0 else 0.0
        
        if self.verbose:
            print(f"  Length statistics (characters):")
            print(f"     Prediction Ø: {avg_pred:.1f}")
            print(f"     Reference Ø:  {avg_ref:.1f}")
            print(f"     Ratio:        {ratio:.2f}")
            
        return {
            'avg_prediction_length': avg_pred,
            'avg_reference_length': avg_ref,
            'length_ratio': ratio
        }

    @staticmethod
    def _normalize_text_for_mauve(text: str) -> str:
        if not text:
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Whitespace normalisieren
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # 3. Trim
        text = text.strip()
        
        return text
    
    def _has_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def evaluate_single(
        self,
        prediction: str,
        reference: str
    ) -> EvaluationMetrics:
        return self.evaluate(
            predictions=[prediction],
            references=[reference]
        )
    
    def batch_evaluate(
        self,
        predictions: List[str],
        references: List[str],
        batch_size: int = 50
    ) -> EvaluationMetrics:

        if self.verbose:
            print(f" Batch evaluation with batch size: {batch_size}")
            print()
        
        return self.evaluate(predictions, references)
    
    def _load_judge_model(self) -> Tuple[Any, Any]:
        try:
            print(f"     Loading Judge model: {self.judge_model_id}")
            print(f"DEBUG: Starting judge model loading...")
            
            # Aggressive cleanup before loading judge model
            print(f"DEBUG: Performing aggressive memory cleanup before loading...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()
            print(f"DEBUG: Memory cleanup complete")
            
            # Quantization Config (4-bit)
            print(f"DEBUG: Creating quantization config...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.judge_model_id)
            
            # Set padding token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.judge_model_id,
                quantization_config=quantization_config,
                device_map={"": 0},  # Explicit: place everything on GPU 0
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            model.eval()
            print(f" DEBUG: Model ready for inference")
            
            return model, tokenizer
        except Exception as e:
            print(f"     Error loading judge model: {e}")
            return None, None

    def _unload_judge_model(self, model, tokenizer):
        print("     Unloading Judge model...")
        
        # Move to CPU first
        if hasattr(model, 'to'):
            model.to('cpu')
        
        del model
        del tokenizer
        
        # Aggressive cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        print("     Judge model unloaded and memory freed")


class ExpertRuleEvaluator:
    """
    Evaluator for expert rules using LLM judge and programmatic checks.
    
    Checks summaries for:
    1. Single-sentence rule (Programmatic)
    2. Structure: Issuer at the beginning (LLM)
    3. Structure: Dispositive verb in 2nd position (LLM)
    4. Structure: Recipient in 3rd position (LLM)
    5. Modern names (LLM)
    6. Forbidden content (LLM)
    """
    
    def __init__(self, llm_predict_fn: Callable[[str], str], verbose: bool = False):
        self.llm_predict_fn = llm_predict_fn
        self.verbose = verbose
        
    def evaluate_summary(self, summary: str) -> Dict[str, Any]:
        results = {}
        
        # 1. LLM Check: Single Sentence (Replaces programmatic check)
        # I have to use the LLM since programmatic checks often fail with abbreviations (e.g. "Heinrich VI.").
        # IMPORTANT: Precise prompt that explicitly asks for counting
        results["violation_single_sentence"] = self._check_llm_boolean(
            summary,
            "Zähle die Anzahl der vollständigen Sätze (getrennt durch Punkt, Fragezeichen oder Ausrufezeichen). "
            "Ignoriere dabei Abkürzungen wie 'z.B.', 'Nr.', 'VI.', 'hl.' oder Datumsangaben. "
            "Enthält der Text GENAU EINEN Satz (nicht mehr, nicht weniger)?",
            expect_yes=True,
            rule_name="single_sentence"
        )
        
        # 2. LLM Check: Structure - Issuer
        results["violation_structure_issuer"] = self._check_llm_boolean(
            summary, 
            "Beginnt die Zusammenfassung mit dem Namen des Ausstellers (z.B. 'Friedrich')?",
            expect_yes=True,
            rule_name="structure_issuer"
        )
        
        # 3. LLM Check: Structure - Verb
        results["violation_structure_verb"] = self._check_llm_boolean(
            summary,
            "Folgt in der Zusammenfassung direkt auf den Namen des Ausstellers ein dispositives Verb (z.B. 'bestätigt', 'schenkt', 'nimmt in Schutz')?",
            expect_yes=True,
            rule_name="structure_verb"
        )
        
        # 4. LLM Check: Structure - Recipient
        results["violation_structure_recipient"] = self._check_llm_boolean(
            summary,
            "Wird in der Zusammenfassung direkt nach dem dispositiven Verb der Empfänger (Begünstigter) genannt?",
            expect_yes=True,
            rule_name="structure_recipient"
        )
        
        # 5. LLM Check: Modern Names
        results["violation_modern_names"] = self._check_llm_boolean(
            summary,
            "Werden in der Zusammenfassung Orts- und Personennamen in ihrer modernen deutschen Form verwendet (soweit identifizierbar) und nicht in der lateinischen Form?",
            expect_yes=True,
            rule_name="modern_names"
        )
        
        # 6. LLM Check: Forbidden Content
        results["violation_forbidden_content"] = self._check_llm_boolean(
            summary,
            "Enthält die Zusammenfassung formelhafte Teile wie Strafandrohungen, Zeugenlisten oder Datierungen, die weggelassen werden sollten?",
            expect_yes=False, # Expect No -> Violation if Yes
            rule_name="forbidden_content"
        )
        
        return results
    
    def evaluate_batch(self, summaries: List[str]) -> Dict[str, float]:
        
        results = []
        for i, s in enumerate(summaries, 1):
            print(f"\n  Sample {i}/{len(summaries)}: \"{s[:80]}...\"")
            result = self.evaluate_summary(s)
            results.append(result)
            
            # Print summary of violations for this sample
            violations = [k for k, v in result.items() if v]
            if violations:
                print(f"     Violations found: {', '.join([k.replace('violation_', '') for k in violations])}")
            else:
                print(f"     No violations")
            
        print(f"\n{'='*70}")
        print(f"  ✓ Expert rule evaluation completed")
        print(f"{'='*70}\n")
        
        n = len(results)
        if n == 0:
            return {}
            
        # Calculate average for all keys starting with "violation_"
        stats = {}
        keys = results[0].keys() if results else []
        
        for key in keys:
            if key.startswith("violation_"):
                stats[f"{key}_rate"] = sum(1 for r in results if r.get(key)) / n
        
        return stats

    def _check_llm_boolean(self, summary: str, question: str, expect_yes: bool, rule_name: str = "") -> bool:
        """Performs a True/False query to the LLM."""
        prompt = f"""Du bist ein strenger, präziser Prüfer für Regesten mittelalterlicher Urkunden.
        Analysiere die folgende Zusammenfassung SORGFÄLTIG und beantworte die Frage.

        Zusammenfassung:
        \"\"\"{summary}\"\"\"

        Frage: {question}

        WICHTIG: Analysiere den Text genau. Antworte NUR mit dem Wort 'true' oder 'false', ohne Erklärung."""
        
        try:
            print(f"🔍 DEBUG: Calling llm_predict_fn for rule {rule_name}...")
            response = self.llm_predict_fn(prompt).strip().lower()
            print(f"🔍 DEBUG: LLM response received for rule {rule_name}: '{response[:100]}'")
            # Entferne Punkt am Ende falls vorhanden
            if response.endswith('.'):
                response = response[:-1]
                
            is_true = "true" in response
            
            if expect_yes:
                is_violation = not is_true # Violation wenn NICHT True
            else:
                is_violation = is_true # Violation wenn True
            
            # Verbose Logging
            if self.verbose:
                status = "VIOLATION" if is_violation else "✓ OK"
                print(f"       [{rule_name}]")
                print(f"         Frage: {question[:80]}...")
                print(f"         LLM-Antwort: '{response}' → {status}")
            
            return is_violation
        except Exception as e:
            print(f"Error during LLM check: {e}")
            return False # In case of doubt, no violation


class ComparisonEvaluator(SummarizationEvaluator):
    def compare(
        self,
        predictions_dict: Dict[str, List[str]],
        references: List[str]
    ) -> Dict[str, EvaluationMetrics]:
        results = {}
        
        if self.verbose:
            print(f"Comparing {len(predictions_dict)} different approaches...")
            print()
        
        for name, predictions in predictions_dict.items():
            if self.verbose:
                print(f"{'='*60}")
                print(f"Approach: {name}")
                print(f"{'='*60}")
            
            metrics = self.evaluate(predictions, references)
            results[name] = metrics
            
            if self.verbose:
                print()
        
        return results
    
    def compare_and_summarize(
        self,
        predictions_dict: Dict[str, List[str]],
        references: List[str]
    ) -> Tuple[Dict[str, EvaluationMetrics], str]:
        results = self.compare(predictions_dict, references)
        
        # Create summary table
        summary_lines = [
            "="*115,
            "COMPARISON OVERVIEW",
            "="*115,
            f"{'Approach':<20} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10} "
            f"{'BLEU':>10} {'BERTScore-F1':>15} {'MAUVE':>10} {'Len-Ratio':>10}",
            "-"*115
        ]
        
        for name, metrics in results.items():
            summary_lines.append(
                f"{name:<20} {metrics.rouge1:>10.4f} {metrics.rouge2:>10.4f} "
                f"{metrics.rougeL:>10.4f} {metrics.bleu_score:>10.2f} "
                f"{metrics.bertscore_f1_avg:>15.4f} {metrics.mauve_score:>10.4f} "
                f"{metrics.length_ratio:>10.2f}"
            )
        
        summary_lines.append("="*115)
        
        # Find best approach per metric
        best_rouge1 = max(results.items(), key=lambda x: x[1].rouge1)
        best_bleu = max(results.items(), key=lambda x: x[1].bleu_score)
        best_bertscore = max(results.items(), key=lambda x: x[1].bertscore_f1_avg)
        best_mauve = max(results.items(), key=lambda x: x[1].mauve_score)
        
        summary_lines.extend([
            "",
            "Best results:",
            f"  • ROUGE-1:       {best_rouge1[0]} ({best_rouge1[1].rouge1:.4f})",
            f"  • BLEU:          {best_bleu[0]} ({best_bleu[1].bleu_score:.2f})",
            f"  • BERTScore-F1:  {best_bertscore[0]} ({best_bertscore[1].bertscore_f1_avg:.4f})",
            f"  • MAUVE:         {best_mauve[0]} ({best_mauve[1].mauve_score:.4f})",
            "="*95
        ])
        
        summary_text = "\n".join(summary_lines)
        
        if self.verbose:
            print(summary_text)
        
        return results, summary_text


class TokenizerEvaluator:
    def __init__(self, tokenizer_id_or_obj: Any):
        if isinstance(tokenizer_id_or_obj, str):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id_or_obj)
            except Exception as e:
                print(f"Could not load tokenizer '{tokenizer_id_or_obj}': {e}")
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer_id_or_obj

    def evaluate(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculates metrics for tokenization.
        """
        if not self.tokenizer or not texts:
            return {}

        total_tokens = 0
        total_chars = 0
        total_single_char_tokens = 0
        total_words = 0
        
        # Batch processing would be faster, but we do it iteratively for more precise analysis
        for text in texts:
            if not text: continue
            
            # Simple word count (whitespace)
            words = text.split()
            total_words += len(words)
            
            # Tokenisierung
            tokens = self.tokenizer.tokenize(text)
            num_tokens = len(tokens)
            total_tokens += num_tokens
            
            # Character count (without whitespace for fair comparison)
            chars_in_text = len(text.replace(" ", ""))
            total_chars += chars_in_text
            
            # Analysis of individual tokens
            for token in tokens:
                # Clean token from typical SentencePiece/BPE markers (e.g.   or Ġ)
                clean_token = token.replace(" ", "").replace("Ġ", "").replace("</w>", "")
                if len(clean_token) == 1:
                    total_single_char_tokens += 1

        if total_tokens == 0:
            return {}

        return {
            "avg_tokens_per_word": total_tokens / total_words if total_words > 0 else 0,
            "avg_chars_per_token": total_chars / total_tokens,
            "single_char_token_rate": total_single_char_tokens / total_tokens
        }


if __name__ == "__main__":
    # Example usage
    print("Evaluator Module Test\n")
    print("="*60)
    
    # Test data
    predictions = [
        # "Friedrich gewährt und bestätigt seinem getreuen Diener Matthäus de Calvello aus Gnade und aufgrund dessen treuer Dienste die Terra Buxemi mit allen dazugehörigen Rechten und Besitzungen auf ewige Zeiten für ihn und seine Erben.",
        "Friedrich befiehlt den Bürgern von Ivrea, die Rechte des Bischofs von Ivrea als Grafen der Stadt zu respektieren und bestimmte Missstände, wie die Erhebung einer salaria, die Beschlagnahme von Geld und die Aufnahme von Leuten, die dem Bischof feindlich gesinnt sind, abzustellen. Bei Zuwiderhandlung droht er mit dem Bann und einer Geldstrafe von 1000 Mark Silber.",
    ]
    
    references = [
        "Friedrich befiehlt den Geschworenen und der Gesamtheit, nicht die Steuerfreiheit der Kleriker der Palastkapelle zu Palermo hinsichtlich ihrer Tiere und Besitzungen zu beeinträchtigen, wie dies seit der Zeit seiner Abreise geschehen ist.",
    ]
    
    # ===== EXPERT RULES TEST (with built-in Judge Model) =====
    # WARNING: This loads the Judge Model (4-bit quantized) and requires GPU!
    
    # Choose the Judge Model:
    # - "Qwen/Qwen2.5-7B-Instruct" (Default, faster)
    # - "Qwen/Qwen3-8B" (better, but slightly slower)
    judge_model = "Qwen/Qwen3-8B"
    
    evaluator = SummarizationEvaluator(
        verbose=True,
        judge_model_id=judge_model
    )
    
    metrics = evaluator.evaluate(
        predictions, 
        references, 
        use_expert_rules=True
    )
    
    print(metrics.to_json())
    
    if metrics.expert_violations:
        print("📋 Expert Rule Violations (Summary):")
        for rule, rate in metrics.expert_violations.items():
            status = "❌" if rate > 0 else "✅"
            print(f"  {status} {rule}: {rate:.0%}")
