import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from datasets import load_dataset, Dataset
from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import load_model as mlx_load_model_only
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import math
from mlx.optimizers import Adam, cosine_decay, clip_grad_norm
import re
import copy
import inspect
import random
import argparse
import tomllib  # Python 3.11+
import pickle
from pathlib import Path

# -------------------------------------------------------------------
# Dataset Preparation and Formatting
# -------------------------------------------------------------------
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Wordle dataset loading
def get_wordle_dataset(split="train") -> Dataset:
    """Load Wordle GRPO dataset and format as chat messages"""
    from reward_functions import output_format_check, uses_previous_feedback, guess_value
    
    data = load_dataset('predibase/wordle-grpo', split=split)  # type: ignore
    
    # Convert to chat format expected by the trainer
    def format_as_chat(x):
        # The prompt is already formatted, we just need to wrap it properly
        # Extract the actual user message (everything after system prompt)
        prompt_text = x['prompt']
        
        # For now, just pass the text directly
        # The trainer will need to handle raw text, not chat format
        return {
            'prompt': prompt_text,  # Keep as string - will be converted in format_prompt
            'secret': x['secret']
        }
    
    data = data.map(format_as_chat)  # type: ignore
    return data  # type: ignore

# -------------------------------------------------------------------
# Reward Functions
# -------------------------------------------------------------------
def wordle_reward_func(prompts, completions, secret, **kwargs) -> list[float]:
    """Compute Wordle rewards for each completion"""
    from reward_functions import output_format_check, uses_previous_feedback, guess_value
    
    responses = [completion[0]['content'] for completion in completions]
    secret_word = secret[0] if isinstance(secret, (list, tuple)) else secret
    
    scores = []
    for i, response in enumerate(responses):
        example = {
            'word_list': 'five_letter_words.csv',
            'past_guess_history': [],
            'secret_word': secret_word
        }
        
        # Compute rewards (these are imported from reward_functions.py)
        format_r = output_format_check("", response, example)
        feedback_r = uses_previous_feedback("", response, example)
        info_r = guess_value("", response, example)
        
        total_reward = format_r + feedback_r + info_r
        scores.append(total_reward)
        
        # Always log first response of each batch for debugging
        if i == 0:
            print(f"\n{'='*60}")
            print(f"SECRET: {secret_word}")
            print(f"RESPONSE:\n{response}")
            print(f"REWARD: {total_reward:.2f} (format={format_r:.2f}, feedback={feedback_r:.2f}, info={info_r:.2f})")
            print(f"{'='*60}\n")
    
    return scores

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    # Require full XML shell; allow arbitrary newlines/whitespace
    pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    score = 0.0
    if re.search(r"<reasoning>.*?</reasoning>", text, flags=re.DOTALL):
        score += 0.25
    if re.search(r"<answer>.*?</answer>", text, flags=re.DOTALL):
        score += 0.25
    # Penalize trailing junk after </answer>
    end = re.search(r"</answer>(.*)$", text, flags=re.DOTALL)
    if end:
        score -= len(end.group(1).strip()) * 0.001
    return score

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# -------------------------------------------------------------------
# Model Configuration and Loading (Pure MLX)
# -------------------------------------------------------------------
# Note: MLX-LM has built-in LoRA support via command-line tools
# For training with LoRA, you can use: python -m mlx_lm.lora --model <model> --train
# This implementation focuses on full model fine-tuning with GRPO

class TiktokenTokenizerWrapper:
    """Wrapper to make tiktoken.Encoding compatible with GRPO expectations"""
    
    def __init__(self, tiktoken_tokenizer):
        self.tiktoken = tiktoken_tokenizer
        # Set up special tokens
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|endoftext|>"
        self.bos_token = "<|endoftext|>"
        
        # Get token IDs (tiktoken may not have these as properties)
        try:
            self.eos_token_id = tiktoken_tokenizer.eot_token
        except (AttributeError, KeyError):
            # If no eot_token, try encoding the token
            self.eos_token_id = tiktoken_tokenizer.encode(self.eos_token, allowed_special="all")[0]
        
        self.pad_token_id = self.eos_token_id
        self.bos_token_id = self.eos_token_id
        
        # Add properties needed by mlx_lm
        self.vocab_size = tiktoken_tokenizer.n_vocab
        self.all_special_tokens = [self.eos_token, self.pad_token, self.bos_token]
        self.all_special_ids = [self.eos_token_id, self.pad_token_id, self.bos_token_id]
        self.chat_template = None  # Nanochat doesn't have a specific chat template
        self.clean_up_tokenization_spaces = True  # Standard tokenizer behavior
        
    def encode(self, text, add_special_tokens=False):
        """Encode text to token IDs"""
        tokens = self.tiktoken.encode(text, allowed_special="all")
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs to text"""
        if isinstance(token_ids, list):
            return self.tiktoken.decode(token_ids)
        else:
            # Handle single token
            return self.tiktoken.decode([token_ids])
    
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        """Apply chat template - simple version for nanochat"""
        # For nanochat, just concatenate messages with simple formatting
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"{content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
        
        if add_generation_prompt:
            formatted += "Assistant: "
        
        if tokenize:
            return self.encode(formatted)
        return formatted
    
    def get_vocab(self):
        """Return vocab dictionary - minimal implementation for compatibility"""
        # For tiktoken, we don't have easy access to the full vocab
        # Return a minimal dict with special tokens that code might check for
        return {
            self.eos_token: self.eos_token_id,
            self.pad_token: self.pad_token_id,
            self.bos_token: self.bos_token_id,
        }

def load_tiktoken_tokenizer(model_path):
    """Load tiktoken tokenizer from pickle file"""
    model_path = Path(model_path)
    tokenizer_pkl = model_path / "tokenizer.pkl"
    
    if not tokenizer_pkl.exists():
        return None
    
    try:
        with open(tokenizer_pkl, "rb") as f:
            tiktoken_tokenizer = pickle.load(f)
        
        print(f"✅ Loaded tiktoken tokenizer (vocab size: {tiktoken_tokenizer.n_vocab})")
        return TiktokenTokenizerWrapper(tiktoken_tokenizer)
    except Exception as e:
        print(f"⚠️  Failed to load tiktoken tokenizer: {e}")
        return None

def load_model(model_name):
    """Load model and tokenizer using MLX-LM, with custom tiktoken support"""
    
    # First, try to load tiktoken tokenizer if available
    tiktoken_tok = load_tiktoken_tokenizer(model_name)
    
    if tiktoken_tok is not None:
        # Load model without tokenizer  
        print(f"Loading model from {model_name}...")
        from mlx_lm.utils import load_config
        
        # Load config and model manually
        model_path = Path(model_name)
        config = load_config(model_path)
        
        # Import the appropriate model class
        from mlx_lm.models import nanochat
        model_args = nanochat.ModelArgs(**{k:v for k,v in config.items() if k in ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'max_position_embeddings', 'intermediate_size', 'rope_theta']})
        model = nanochat.Model(model_args)
        
        # Load weights
        weights = mx.load(str(model_path / "model.safetensors"))
        model.load_weights(list(weights.items()), strict=False)
        
        print(f"✅ Model loaded with tiktoken tokenizer!")
        return model, tiktoken_tok
    
    # Fall back to standard mlx_load
    print(f"Loading model with standard tokenizer...")
    model, tokenizer = mlx_load(model_name, tokenizer_config={"trust_remote_code": True})

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def calculate_log_probs_single(model, tokenizer, prompt: str, completion: str) -> mx.array:
    """Return ``log p(o_i | q)`` for a single completion.

    The article computes the likelihood ratio between the trainable policy and
    the frozen rollout policy using complete reasoning traces.  To mirror that
    behaviour we feed the concatenated prompt + completion through the model
    and sum the token log probabilities for the completion span only.  The
    helper works for either ``nn.Module`` models or the dictionary-wrapped
    format returned by :mod:`mlx_lm`.
    """
    # Tokenize prompt and completion separately to know boundaries
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    # Create full sequence
    full_tokens = prompt_tokens + completion_tokens
    # Be explicit about dtype: embeddings consume integer IDs.
    input_ids = mx.array(full_tokens, dtype=mx.int32)[None, :]  # Add batch dimension

    # Forward pass through the supplied model to obtain token logits.
    if isinstance(model, nn.Module):
        logits = model(input_ids)
    elif isinstance(model, dict) and 'model' in model:
        logits = model['model'](input_ids)
    else:
        raise ValueError(f"Unexpected model type: {type(model)}")

    # Convert logits into log-probabilities over the vocabulary for every
    # timestep.  ``nn.log_softmax`` provides the numerically stable
    # implementation used throughout the GRPO derivation.
    log_probs_full = nn.log_softmax(logits, axis=-1)

    # Extract log probs for completion tokens
    # log_probs_full[i] predicts token at position i+1
    prompt_len = len(prompt_tokens)
    completion_len = len(completion_tokens)

    # Extract log-probabilities that correspond to the completion tokens only.
    completion_log_probs = []
    for i in range(completion_len):
        pos = prompt_len - 1 + i  # Position in sequence
        if pos < len(full_tokens) - 1:
            next_token_id = full_tokens[pos + 1]
            log_prob = log_probs_full[0, pos, next_token_id]
            completion_log_probs.append(log_prob)

    # Sum the completion log-probabilities to obtain log p(o_i | q).
    if len(completion_log_probs) > 0:
        return mx.sum(mx.stack(completion_log_probs))
    else:
        return mx.array(0.0)

# -------------------------------------------------------------------
# Initialize and Run GRPO Training (Pure MLX)
# -------------------------------------------------------------------
@dataclass
class MLXGRPOConfig:
    """Configuration class for MLX GRPO training"""
    # Core run metadata
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "outputs/Qwen-1.5B-MLX-GRPO"
    run_name: str = "Qwen-1.5B-MLX-GRPO-gsm8k"
    learning_rate: float = 1e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_train_samples: int = 0  # Limit training samples (0 = use all)
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.1
    logging_steps: int = 1
    num_generations: int = 64  # DeepSeekMath uses 64 samples per prompt
    max_prompt_length: int = 512
    max_completion_length: int = 1024  # DeepSeekMath uses 1024
    max_new_tokens: int = 512
    temperature: float = 0.7
    clip_eps: float = 0.2  # PPO clipping epsilon
    kl_coeff: float = 0.0  # KL coefficient (can set to 0.04 as in DeepSeek)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    lr_scheduler_type: str = 'cosine'
    save_steps: int = 100
    eval_steps: int = 50  # Run EM evaluation every N steps
    eval_samples: int = 200  # Number of samples to use for evaluation
    seed: int = 0
    use_compile: bool = True  # Toggle mx.compile for gradient computation
    quantize_for_rollouts: bool = True  # Quantize model_old/ref_model to 4-bit (disable for large models)
    # --- evaluation & logging ---
    eval_every_updates: int = 25       # set 0 to disable periodic eval
    eval_subset_size: int = 200
    eval_max_new_tokens: int = 128
    log_jsonl: bool = True

# -------------------------
# Config helpers
# -------------------------
def _coerce_value(val: str, target):
    """Best-effort string -> target type coercion."""
    t = target
    if t is bool:
        return val.lower() in {"1", "true", "yes", "on"}
    if t is int:
        return int(val)
    if t is float:
        return float(val)
    return val  # str or anything else: leave as-is

def load_toml_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)

def update_config_from_dict(cfg: MLXGRPOConfig, d: Dict[str, Any]) -> MLXGRPOConfig:
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

def apply_overrides(cfg: MLXGRPOConfig, overrides: list[str]) -> MLXGRPOConfig:
    """Override fields with --set key=value (repeatable)."""
    hints = MLXGRPOConfig.__annotations__
    for item in overrides:
        if "=" not in item:
            print(f"[warn] ignoring override (no '='): {item}")
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not hasattr(cfg, key):
            print(f"[warn] unknown config key: {key}")
            continue
        target_type = hints.get(key, str)
        try:
            coerced = _coerce_value(val, target_type)
        except Exception:
            print(f"[warn] failed to coerce '{val}' to {target_type}; using string")
            coerced = val
        setattr(cfg, key, coerced)
    return cfg

class MLXGRPOTrainer:
    def __init__(self, model, tokenizer, reward_funcs, args: MLXGRPOConfig, train_dataset, eval_dataset=None):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # CRITICAL: Three models as per article
        # 1. model - current trainable policy (π_θ)
        # 2. model_old - policy that generates responses (π_θ_old), synced periodically
        # 3. ref_model - original pretrained model (π_ref), never updated
        self.model_old = copy.deepcopy(model)  # πθ_old
        self.ref_model = copy.deepcopy(model)  # π_ref
        
        # Optionally quantize for faster rollouts (can cause instability in large models)
        if args.quantize_for_rollouts:
            try:
                # Safe: these models are not trained
                nn.quantize(self.model_old, group_size=64, bits=4)
                nn.quantize(self.ref_model, group_size=64, bits=4)
                print("Quantized model_old and ref_model to 4-bit for faster rollouts.")
            except Exception as e:
                print(f"Quantization skipped: {e}")
        else:
            print("Quantization disabled (quantize_for_rollouts=False). Using full precision for rollouts.")

        # Steps / updates accounting
        self.step = 0                                    # batch steps (for logs)
        self.total_batches = len(train_dataset)
        self.updates_per_epoch = max(1, math.ceil(self.total_batches / args.gradient_accumulation_steps))
        self.total_updates = self.updates_per_epoch * args.num_epochs
        self.update_every = 10                           # Sync model_old every N *batch* steps

        # LR schedule with warmup + cosine, defined over *optimizer updates*
        base = cosine_decay(args.learning_rate, self.total_updates)
        warmup_steps = max(1, int(self.total_updates * self.args.warmup_ratio))
        def schedule(step: int):
            warm = step / warmup_steps if step < warmup_steps else 1.0
            return base(step) * warm
        self.lr_schedule = schedule
        self.optimizer = Adam(learning_rate=self.lr_schedule)

        # Gradient accumulation
        self._accum_grads = None
        # Optimizer update counter (distinct from batch steps)
        self.update_step = 0
        # Logging path
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.log_path = os.path.join(self.args.output_dir, "training_log.jsonl")

        # Tracking for logging
        self.last_reward_mean = 0.0
        self.last_reward_std = 0.0
        self.last_em_score = None

    def format_prompt(self, messages) -> str:
        """
        Build the prompt - handles both chat format and raw strings for Wordle.
        """
        # If it's already a string (Wordle format), just add assistant marker
        if isinstance(messages, str):
            return messages + "\n<|im_start|>assistant\n"
        
        # Otherwise, use chat template
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            formatted = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    formatted += f"System: {content}\n\n"
                elif role == 'user':
                    formatted += f"User: {content}\n\n"
                elif role == 'assistant':
                    formatted += f"Assistant: {content}\n\n"
            formatted += "Assistant: "
            return formatted

    # -------------------------
    # JSONL logger (append)
    # -------------------------
    def _log_jsonl(self, record: Dict[str, Any]):
        if not self.args.log_jsonl:
            return
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"Logging failed: {e}")

    def generate_responses(self, batch):
        """
        Generate multiple responses for a prompt using model_old.
        Also computes old_log_probs for each response.

        Returns:
            responses: List of generated strings
            old_log_probs: Array of log probability values (mx.array)
            formatted_prompt: The formatted prompt string
        """
        messages = batch['prompt']
        formatted_prompt = self.format_prompt(messages)
        
        # Nudge the model to follow the required format
        formatted_prompt = formatted_prompt + "<reasoning>"

        responses: List[str] = []
        old_log_probs: List[mx.array] = []

        # Sampler & processors (temperature/top-p via sampler)
        sampler = make_sampler(self.args.temperature, top_p=0.95, min_p=0.0, min_tokens_to_keep=1)
        logits_processors = make_logits_processors(
            None, repetition_penalty=None, repetition_context_size=None
        )

        # Try batch generation for better performance
        # Note: mlx_generate may support batch generation depending on version
        try:
            # Attempt batch generation with repeated prompts
            prompts = [formatted_prompt] * self.args.num_generations
            outputs = mlx_generate(
                self.model_old,
                self.tokenizer,
                prompt=prompts,
                max_tokens=self.args.max_new_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                verbose=False
            )
            # Check if outputs is iterable (batch generation succeeded)
            if isinstance(outputs, (list, tuple)):
                # Trim at the end tag if present to reduce junk after </answer>
                responses = []
                for out in outputs:
                    cut = out
                    if "</answer>" in cut:
                        cut = cut.split("</answer>", 1)[0] + "</answer>"
                    responses.append(cut)
            else:
                # Single output, fall back to loop
                raise TypeError("Batch generation not supported")

            # Compute log probs for all responses
            old_log_probs = [
                calculate_log_probs_single(self.model_old, self.tokenizer, formatted_prompt, out)
                for out in responses
            ]
        except (TypeError, Exception) as e:
            # Fall back to sequential generation
            responses = []
            old_log_probs = []
            for i in range(self.args.num_generations):
                try:
                    # CRITICAL: Generate with model_old (not self.model)
                    output = mlx_generate(
                        self.model_old,  # Use old model for generation!
                        self.tokenizer,
                        prompt=formatted_prompt,
                        max_tokens=self.args.max_new_tokens,
                        sampler=sampler,
                        logits_processors=logits_processors,
                        verbose=False,
                    )
                    # Trim at the end tag if present to reduce junk after </answer>
                    cut = output
                    if "</answer>" in cut:
                        cut = cut.split("</answer>", 1)[0] + "</answer>"
                    responses.append(cut)

                    # CRITICAL: Compute old_log_probs during generation
                    # This is π_θ_old(o_i|q) - probability under the old policy
                    log_prob = calculate_log_probs_single(
                        self.model_old,
                        self.tokenizer,
                        formatted_prompt,
                        cut
                    )
                    old_log_probs.append(log_prob)

                except Exception as e:
                    print(f"Generation {i} failed: {e}")
                    # Fallback to empty response with zero log prob
                    responses.append("")
                    old_log_probs.append(mx.array(0.0))

        # Stack the rollout log-probabilities into a single tensor to align with
        # the vectorised GRPO loss calculation.
        if len(old_log_probs) > 0:
            old_log_probs_arr = mx.stack(old_log_probs)
        else:
            old_log_probs_arr = mx.array([])

        return responses, old_log_probs_arr, formatted_prompt
    
    def compute_rewards(self, batch, responses: List[str]) -> mx.array:
        """
        Compute rewards for all responses using reward functions.
        Returns normalized advantages based on group mean.
        """
        if len(responses) == 0:
            return mx.zeros((0,)), mx.zeros((0,))

        # Prepare completions in the format expected by reward functions.  Each
        # completion is wrapped in the structure ``[{"content": text}]`` that
        # the existing reward utilities consume.
        completions = [[{"content": response}] for response in responses]

        # Accumulate rewards from every reward function defined for the run.
        total_rewards = mx.zeros((len(responses),))
        reward_context = {
            "prompts": [batch["prompt"]],
            "answer": [batch.get("answer", "")],
            "secret": [batch.get("secret", "")],  # Add secret for Wordle
        }

        for reward_fn in self.reward_funcs:
            try:
                # Inspect the callable signature to determine whether the
                # reward expects the ``prompts`` or ``answer`` keyword
                # arguments.  Functions defined in this file are permissive and
                # accept **kwargs, but the check keeps the trainer robust if a
                # custom reward omits them.
                sig = inspect.signature(reward_fn)
                kwargs: Dict[str, Any] = {}
                if "prompts" in sig.parameters:
                    kwargs["prompts"] = reward_context["prompts"]
                if "answer" in sig.parameters:
                    kwargs["answer"] = reward_context["answer"]
                if "secret" in sig.parameters:  # Add secret support
                    kwargs["secret"] = reward_context["secret"]

                reward_values = reward_fn(completions=completions, **kwargs)

                if not isinstance(reward_values, (list, tuple)):
                    reward_values = [reward_values] * len(responses)

                # Convert the return value into an array so that we can combine
                # contributions from multiple reward functions.
                reward_array = mx.array(reward_values)
                total_rewards = total_rewards + reward_array
            except Exception as e:
                print(f"Reward function failed: {e}")
                continue

        # GRPO advantage normalisation: subtract the group mean and divide by
        # the group standard deviation to obtain ``A_i``.
        mean_reward = mx.mean(total_rewards)
        std_reward = mx.std(total_rewards)
        advantages = (total_rewards - mean_reward) / (std_reward + 1e-8)

        return advantages, total_rewards

    # -------------------------
    # Quick EM evaluator
    # -------------------------
    def evaluate_em(self, dataset, num_samples: int) -> float:
        """Exact‑match on a small subset of GSM8K using the current policy."""
        if num_samples <= 0:
            return 0.0
        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        subset = idxs[:min(num_samples, len(dataset))]

        # Greedy-ish sampler (temperature 0.0)
        sampler = make_sampler(0.0, top_p=1.0, min_p=0.0, min_tokens_to_keep=1)
        logits_processors = make_logits_processors(None, repetition_penalty=None, repetition_context_size=None)

        def maybe_int(s: Optional[str]):
            if s is None:
                return None
            try:
                return int(s.strip())
            except Exception:
                return None

        correct = 0
        total = 0
        for i in subset:
            ex = dataset[i]
            gold = ex.get("answer", None)
            if gold is None:
                continue
            messages = ex["prompt"]
            try:
                prompt = self.format_prompt(messages)
                out = mlx_generate(
                    self.model, self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.args.eval_max_new_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    verbose=False,
                )
                # Trim after </answer> if present
                if "</answer>" in out:
                    out = out.split("</answer>", 1)[0] + "</answer>"
                pred = extract_xml_answer(out)
                # Numeric‑aware EM
                gi, pi = maybe_int(gold), maybe_int(pred)
                if gi is not None and pi is not None:
                    match = (gi == pi)
                else:
                    match = (pred.strip() == gold.strip())
                correct += int(match)
                total += 1
            except Exception as e:
                # Skip problematic samples in eval
                continue
        return (correct / total) if total > 0 else 0.0

    def save_checkpoint(self, path: str):
        """Save model checkpoint with all necessary files for inference"""
        os.makedirs(path, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Saving checkpoint to: {path}")
        print(f"{'='*60}")
        
        # 1. Save model weights (.safetensors via Module.save_weights)
        try:
            if isinstance(self.model, nn.Module):
                self.model.save_weights(os.path.join(path, "model.safetensors"))
            elif isinstance(self.model, dict) and 'model' in self.model:
                self.model['model'].save_weights(os.path.join(path, "model.safetensors"))
            print(f"✓ Model weights saved")
        except Exception as e:
            print(f"✗ Failed to save model weights: {e}")
            return  # Critical failure - can't continue

        # 2. Copy tokenizer files (needed for inference)
        import shutil
        tokenizer_files = [
            'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
            'vocab.json', 'merges.txt', 'added_tokens.json', 'chat_template.jinja'
        ]
        
        # Determine source directory (where model was loaded from)
        source_dir = self.args.model_name
        if not os.path.isdir(source_dir):
            # Model was loaded from HF, look in cache or current dir
            if os.path.exists('tokenizer.json'):
                source_dir = '.'
        
        tokenizer_copied = 0
        for filename in tokenizer_files:
            src_path = os.path.join(source_dir, filename)
            if os.path.exists(src_path):
                try:
                    shutil.copy2(src_path, os.path.join(path, filename))
                    tokenizer_copied += 1
                except Exception:
                    pass
        
        if tokenizer_copied > 0:
            print(f"✓ Copied {tokenizer_copied} tokenizer files")
        else:
            print(f"⚠ No tokenizer files found (model may not load correctly)")

        # 3. Save model config (needed for loading)
        try:
            config_src = os.path.join(source_dir, 'config.json')
            if os.path.exists(config_src):
                shutil.copy2(config_src, os.path.join(path, 'config.json'))
                print(f"✓ Model config saved")
            else:
                print(f"⚠ config.json not found")
        except Exception as e:
            print(f"⚠ Could not copy config: {e}")

        # 4. Save optimizer state (optional - only for resuming training)
        try:
            if hasattr(self.optimizer, "state") and self.optimizer.state:
                mx.save_safetensors(os.path.join(path, "optimizer.safetensors"), self.optimizer.state)
                print(f"✓ Optimizer state saved")
        except Exception as e:
            print(f"⚠ Skipping optimizer state (known MLX issue)")

        # 5. Save training metadata
        try:
            training_state = {
                "step": int(self.step),
                "update_step": int(self.update_step),
                "epoch": 0,
                "model_name": self.args.model_name,
                "args": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
                        for k, v in self.args.__dict__.items()},
            }
            with open(os.path.join(path, "trainer_state.json"), "w") as f:
                json.dump(training_state, f, indent=2)
            print(f"✓ Training metadata saved")
        except Exception as e:
            print(f"⚠ Could not save training metadata: {e}")
        
        print(f"{'='*60}")
        print(f"✅ Checkpoint saved successfully!")
        print(f"{'='*60}\n")

    def compute_grpo_loss(self, policy_model, ref_model, prompt: str,
                          responses: List[str], advantages: mx.array,
                          old_log_probs: mx.array):
        """
        Compute GRPO loss following the article's implementation.

        Args:
            prompt: The formatted prompt string
            responses: List of generated completions
            advantages: Normalized advantages (A_i)
            old_log_probs: Pre-computed log probs from model_old

        Returns:
            loss: Total loss (to be minimized)
            policy_reward_mean: Mean policy reward (for logging)
            kl_div_mean: Mean KL divergence (for logging)
        """
        if len(responses) == 0:
            zero = mx.array(0.0)
            return zero, zero, zero

        # Evaluate log probabilities for each completion under the current
        # policy and the frozen reference model.  The operations remain in the
        # autograd graph so gradients flow back to ``policy_model``.
        current_log_probs = []
        ref_log_probs = []
        for response in responses:
            try:
                current_log_probs.append(
                    calculate_log_probs_single(policy_model, self.tokenizer, prompt, response)
                )
                ref_log_probs.append(
                    calculate_log_probs_single(ref_model, self.tokenizer, prompt, response)
                )
            except Exception as e:
                print(f"Log prob computation failed: {e}")
                current_log_probs.append(mx.array(0.0))
                ref_log_probs.append(mx.array(0.0))

        current_log_probs_arr = mx.stack(current_log_probs)
        ref_log_probs_arr = mx.stack(ref_log_probs)
        advantages_arr = mx.array(advantages)
        old_log_probs_arr = mx.array(old_log_probs)

        # PPO-clip objective: ratio = π_θ / π_θ_old, clipped to the trust region
        # [1-ε, 1+ε].  ``advantages`` acts as the fixed baseline-corrected signal.
        ratio = mx.exp(current_log_probs_arr - old_log_probs_arr)
        clipped_ratio = mx.clip(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps)
        policy_rewards = mx.minimum(ratio * advantages_arr, clipped_ratio * advantages_arr)

        # KL penalty: D_KL(π_θ || π_ref) = r - log(r) - 1, with r = π_ref / π_θ.
        log_ratio_for_kl = ref_log_probs_arr - current_log_probs_arr
        ratio_for_kl = mx.exp(log_ratio_for_kl)
        kl_divs = ratio_for_kl - log_ratio_for_kl - 1

        policy_reward_mean = mx.mean(policy_rewards)
        kl_div_mean = mx.mean(kl_divs)

        # Total objective: maximize (policy_reward - β * KL).  We return the
        # negated mean so that optimisers can minimise it directly.
        objective = policy_rewards - self.args.kl_coeff * kl_divs
        loss = -mx.mean(objective)

        return loss, policy_reward_mean, kl_div_mean

    def evaluate(self) -> float:
        """
        Run exact-match evaluation on a subset of test examples.
        Returns EM score (0-1).
        """
        if self.eval_dataset is None:
            return 0.0

        # Sample eval_samples examples (or use all if fewer)
        eval_size = min(self.args.eval_samples, len(self.eval_dataset))
        eval_indices = random.sample(range(len(self.eval_dataset)), eval_size)

        correct = 0
        total = 0

        # Sampler & processors for evaluation (greedy, temp=0)
        sampler = make_sampler(0.0, top_p=1.0, min_p=0.0, min_tokens_to_keep=1)
        logits_processors = make_logits_processors(
            None, repetition_penalty=None, repetition_context_size=None
        )
        # numeric-aware compare helper (same as evaluate_em)
        def maybe_int(s: Optional[str]):
            if s is None:
                return None
            try:
                return int(s.strip())
            except Exception:
                return None

        for idx in eval_indices:
            example = self.eval_dataset[idx]
            messages = example['prompt']
            gold_answer = example.get('answer', '')

            if gold_answer is None:
                continue

            formatted_prompt = self.format_prompt(messages)

            try:
                # Generate with current model (not model_old)
                output = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=self.args.eval_max_new_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    verbose=False,
                )
                # Trim after </answer> if present
                if "</answer>" in output:
                    output = output.split("</answer>", 1)[0] + "</answer>"
                predicted = extract_xml_answer(output).strip()
                gold_str = gold_answer.strip()
                # numeric-aware EM
                gi, pi = maybe_int(gold_str), maybe_int(predicted)
                match = (gi == pi) if (gi is not None and pi is not None) else (predicted == gold_str)
                correct += int(match)
                total += 1

            except Exception as e:
                print(f"Eval generation failed: {e}")
                continue

        em_score = correct / total if total > 0 else 0.0
        self.last_em_score = em_score
        print(f"\n*** Evaluation: {correct}/{total} correct (EM: {em_score:.2%}) ***")
        return em_score

    def train_step(self, batch):
        """
        Performs a single training step using GRPO (following the article).

        Steps:
        1. Generate N responses using model_old and compute old_log_probs
        2. Compute rewards and advantages (relative to group mean)
        3. Compute GRPO loss (uses model, ref_model, and old_log_probs)
        4. Update model parameters
        5. Periodically sync model_old from model
        """
        next_step = self.step + 1
        print(f"\n{'='*60}")
        print(f"Training step {next_step}")
        print(f"{'='*60}")

        # 1. Generate multiple responses with model_old
        responses, old_log_probs, formatted_prompt = self.generate_responses(batch)
        num_responses = len(responses)
        print(f"Generated {num_responses} responses using model_old")

        if num_responses == 0:
            print("No responses were produced; skipping update.")
            return mx.array(0.0), mx.array(0.0), mx.array(0.0)

        # 2. Compute rewards and advantages
        advantages, rewards = self.compute_rewards(batch, responses)
        self.last_reward_mean = float(mx.mean(rewards)) if num_responses > 0 else 0.0
        self.last_reward_std = float(mx.std(rewards)) if num_responses > 0 else 0.0
        adv_mean = float(mx.mean(advantages)) if num_responses > 0 else 0.0
        adv_std = float(mx.std(advantages)) if num_responses > 0 else 0.0
        print(f"Rewards - Mean: {self.last_reward_mean:.3f}, Std: {self.last_reward_std:.3f}")
        print(f"Advantages - Mean: {adv_mean:.3f}, Std: {adv_std:.3f}")

        # Display sample response
        if len(responses) > 0:
            print(f"\n--- Sample Response (Reward: {float(rewards[0]):.3f}) ---")
            print(responses[0][:200] + "..." if len(responses[0]) > 200 else responses[0])
            print(f"---")

        # 3. Define loss function for gradient computation
        def loss_fn():
            """Return only the scalar loss; compute metrics separately."""
            loss, _, _ = self.compute_grpo_loss(
                self.model,
                self.ref_model,
                formatted_prompt,
                responses,
                advantages,
                old_log_probs,
            )
            return loss

        # 4. Compute loss and gradients
        try:
            # Compute gradients (with optional compilation). No aux in this MLX.
            grad_fn_raw = nn.value_and_grad(self.model, loss_fn)
            loss_and_grad_fn = mx.compile(grad_fn_raw) if self.args.use_compile else grad_fn_raw
            loss, grads = loss_and_grad_fn()

            # Recompute metrics for logging (no grad needed here)
            policy_reward, kl_div = mx.array(0.0), mx.array(0.0)
            try:
                _, policy_reward, kl_div = self.compute_grpo_loss(
                    self.model,
                    self.ref_model,
                    formatted_prompt,
                    responses,
                    advantages,
                    old_log_probs,
                )
            except Exception:
                pass

            # Accumulate grads
            if self._accum_grads is None:
                self._accum_grads = grads
            else:
                self._accum_grads = tree_map(lambda a, b: a + b, self._accum_grads, grads)

            do_update = ((self.step + 1) % self.args.gradient_accumulation_steps) == 0
            eval_items = []
            if do_update:
                # Scale, clip, update
                scaled = tree_map(lambda g: g / self.args.gradient_accumulation_steps, self._accum_grads)
                scaled, grad_norm = clip_grad_norm(scaled, self.args.max_grad_norm)
                self.optimizer.update(self.model, scaled)
                self._accum_grads = None
                # collect items to eval when we actually update
                if isinstance(self.model, nn.Module):
                    eval_items.append(self.model.trainable_parameters())
                elif isinstance(self.model, dict) and 'model' in self.model:
                    eval_items.append(self.model['model'].trainable_parameters())
                if hasattr(self.optimizer, "state"):
                    eval_items.append(self.optimizer.state)
                if eval_items:
                    mx.eval(*eval_items)

            if do_update:
                print(
                    f"Loss: {float(loss):.4f}, "
                    f"GradNorm: {float(grad_norm):.4f}, "
                    f"Policy Reward: {float(policy_reward):.4f}, KL: {float(kl_div):.4f}"
                )
                # Bump optimizer update counter and log
                self.update_step += 1
                lr_now = float(self.lr_schedule(self.update_step))
                # Log JSONL
                self._log_jsonl({
                    "update": int(self.update_step),
                    "batch_step": int(self.step + 1),
                    "lr": lr_now,
                    "loss": float(loss),
                    "grad_norm": float(grad_norm),
                    "policy_reward": float(policy_reward),
                    "kl": float(kl_div),
                    "reward_mean": float(self.last_reward_mean),
                    "reward_std": float(self.last_reward_std),
                })
                # Periodic EM evaluation on a small subset
                if self.args.eval_every_updates > 0 and (self.update_step % self.args.eval_every_updates == 0):
                    em = self.evaluate_em(self.train_dataset, self.args.eval_subset_size)
                    print(f"[Eval] EM@{self.args.eval_subset_size}: {em:.3f}")
                    self._log_jsonl({
                        "update": int(self.update_step),
                        "em_subset": int(self.args.eval_subset_size),
                        "em": float(em),
                    })

        except Exception as e:
            print(f"Training step failed: {e}")
            import traceback
            traceback.print_exc()
            loss = mx.array(0.0)
            policy_reward = mx.array(0.0)
            kl_div = mx.array(0.0)

        self.step += 1

        # 6. Sync model_old from model periodically (as per article)
        if self.step % self.update_every == 0:
            print(f"\n*** Syncing model_old weights from model (step {self.step}) ***")
            # Copy current model to model_old
            if isinstance(self.model, nn.Module):
                self.model_old = copy.deepcopy(self.model)
            elif isinstance(self.model, dict):
                self.model_old = copy.deepcopy(self.model)
            # Re-quantize rollout policy to keep memory/speed benefits
            try:
                nn.quantize(self.model_old, group_size=64, bits=4)
            except Exception as e:
                print(f"Re-quantization of model_old skipped: {e}")
            mx.eval(self.model_old)  # Ensure copy is complete

        return loss, policy_reward, kl_div

    def train(self):
        """Enhanced training loop with proper logging and checkpointing"""
        print(f"Starting training: {self.total_batches} batches/epoch × {self.args.num_epochs} epochs = {self.total_updates} optimizer updates")
        print(f"Logging metrics to {self.log_path}")

        for epoch in range(self.args.num_epochs):
            indices = list(range(len(self.train_dataset)))
            random.shuffle(indices)
            
            # Limit training samples if specified
            if self.args.max_train_samples > 0:
                indices = indices[:self.args.max_train_samples]
                print(f"[INFO] Limiting training to {len(indices)} samples (max_train_samples={self.args.max_train_samples})")

            for idx in indices:
                batch = self.train_dataset[idx]
                # Each batch corresponds to one prompt/answer pair.  Iterating
                # through a shuffled epoch mimics the expectation over prompts
                # discussed in the GRPO article.
                # Training step
                loss, policy_reward, kl_div = self.train_step(batch)

                # (Per-update JSONL logging already handled in train_step via _log_jsonl)

                # Logging
                if self.step % self.args.logging_steps == 0:
                    current_update = self.step // self.args.gradient_accumulation_steps
                    print(
                        "Epoch {epoch}, Batch {step}/{total_batches}, Update {update}/{total_updates}, "
                        "Loss: {loss:.4f}, Policy Reward: {pr:.4f}, KL: {kl:.4f}".format(
                            epoch=epoch,
                            step=self.step,
                            total_batches=self.total_batches * self.args.num_epochs,
                            update=current_update,
                            total_updates=self.total_updates,
                            loss=float(loss),
                            pr=float(policy_reward),
                            kl=float(kl_div)
                        )
                    )
                
                # Run evaluation
                if self.step > 0 and self.step % self.args.eval_steps == 0:
                    em_score = self.evaluate()

                # Save checkpoint
                if self.step % self.args.save_steps == 0:
                    checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint-{self.step}")
                    self.save_checkpoint(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

        print(f"\nTraining log saved to {self.log_path}")

def main():
    """Main training function"""
    # ---------------- CLI / Config ----------------
    parser = argparse.ArgumentParser(description="MLX-GRPO trainer for Wordle")
    parser.add_argument(
        "--config",
        default=os.environ.get("MLX_GRPO_CONFIG", "configs/wordle_grpo.toml"),
        help="Path to a TOML config (default: configs/wordle_grpo.toml or $MLX_GRPO_CONFIG).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config keys as --set key=value (repeatable).",
    )
    args = parser.parse_args()

    # Start with defaults, then load TOML if it exists, then apply overrides
    config = MLXGRPOConfig()
    if os.path.exists(args.config):
        print(f"[config] loading {args.config}")
        toml_cfg = load_toml_config(args.config)
        # support either flat keys or a [mlx_grpo] table
        flat = toml_cfg.get("mlx_grpo", toml_cfg)
        config = update_config_from_dict(config, flat)
    else:
        print(f"[config] file not found, using defaults: {args.config}")
    if args.set:
        print(f"[config] applying overrides: {args.set}")
        config = apply_overrides(config, args.set)

    # Seed for reproducibility
    mx.random.seed(config.seed)
    random.seed(config.seed)

    # Resolve per-run output directory early to avoid clobbering
    run_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    config.output_dir = run_dir

    print("="*80)
    print("MLX-GRPO Training Pipeline")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Output Dir: {config.output_dir}")
    print(f"Dataset: GSM8K (train split)")
    print("="*80)

    # Persist the resolved configuration for reproducibility
    with open(os.path.join(config.output_dir, "config.resolved.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\nTraining Configuration:")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Generations per Prompt: {config.num_generations}")
    print(f"  Max New Tokens: {config.max_new_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Clip Epsilon: {config.clip_eps}")
    print(f"  KL Coefficient: {config.kl_coeff}")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    dataset = get_wordle_dataset(split='train')
    eval_dataset = get_wordle_dataset(split='train')  # Wordle only has train split
    print(f"Train dataset loaded: {len(dataset)} examples")
    print(f"Eval dataset loaded: {len(eval_dataset)} examples")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model(config.model_name)
    print(f"Model loaded successfully")
    print(f"Model type: {type(model)}")

    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = MLXGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[
            wordle_reward_func,  # Single comprehensive reward function
        ],
        args=config,
        train_dataset=dataset,
        eval_dataset=eval_dataset
    )
    print("Trainer initialized")

    # Start training
    print("\n" + "="*80)
    print("Starting GRPO Training")
    print("="*80)
    trainer.train()

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Model saved to: {config.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
