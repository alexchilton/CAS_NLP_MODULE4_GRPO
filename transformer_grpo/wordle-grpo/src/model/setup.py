"""
Model loading and setup utilities.

This module provides functions to load, configure, and prepare models
for GRPO training, including quantization, LoRA setup, and checkpointing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

from utils.device import get_device, is_cuda_available

logger = logging.getLogger(__name__)

# Conditional import for bitsandbytes (not available on Mac)
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("bitsandbytes not available - quantization will be disabled")


def load_model_and_tokenizer(
    config: Any,
    device: Optional[torch.device] = None,
    model_dir: Optional[Union[str, Path]] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer.

    If model_dir is provided, loads a fine-tuned model from a local directory.
    Otherwise, loads a base model from HuggingFace with optional quantization and LoRA.
    """
    if model_dir:
        return _load_local_model(config, device, model_dir)
    else:
        return _load_hf_model(config, device)

def _load_local_model(
    config: Any,
    device: Optional[torch.device] = None,
    model_dir: Union[str, Path] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a fine-tuned model from a local directory."""
    if device is None:
        device = get_device()
    
    model_dir = Path(model_dir)
    logger.info(f"Loading fine-tuned model from: {model_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # Load base model
    base_model = _load_standard_model(config.model.name, device)

    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_dir)
    
    logger.info("Fine-tuned model loaded successfully.")
    return model, tokenizer

def _load_hf_model(
    config: Any,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a base model from HuggingFace with optional quantization and LoRA."""
    if device is None:
        device = get_device()

    model_name = config.model.name
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = _load_tokenizer(model_name)

    # Determine if we should use quantization
    use_quantization = config.model.use_quantization and is_cuda_available() and BITSANDBYTES_AVAILABLE

    if config.model.use_quantization and not use_quantization:
        logger.warning(
            "Quantization requested but not available "
            "(requires CUDA and bitsandbytes). Loading without quantization."
        )

    # Load model with optional quantization
    if use_quantization:
        logger.info("Loading model with 4-bit quantization")
        model = _load_quantized_model(model_name)
    else:
        logger.info("Loading model without quantization")
        model = _load_standard_model(model_name, device)

    # Apply LoRA
    model = _apply_lora(model, config, use_quantization)

    # Log model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model loaded: {trainable_params:,} trainable / {total_params:,} total parameters "
        f"({100 * trainable_params / total_params:.2f}% trainable)"
    )

    return model, tokenizer


def _load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load and configure tokenizer.

    Args:
        model_name: Name of the model on HuggingFace.

    Returns:
        Configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",  # Important for batch generation
    )

    # Add special tokens if needed
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # For chat models, ensure chat template is available
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is None:
        logger.info("No chat template found, will use standard prompting")

    logger.info(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

    return tokenizer


def _load_standard_model(model_name: str, device: torch.device) -> PreTrainedModel:
    """
    Load model without quantization.

    Args:
        model_name: Name of the model on HuggingFace.
        device: Target device.

    Returns:
        Loaded model.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32 if device.type == "cpu" else torch.float16,
        device_map=None,  # We'll move to device manually
    )

    # Move to device
    model = model.to(device)
    logger.info(f"Model moved to {device}")

    return model


def _load_quantized_model(model_name: str) -> PreTrainedModel:
    """
    Load model with 4-bit quantization using bitsandbytes.

    Args:
        model_name: Name of the model on HuggingFace.

    Returns:
        Quantized model.
    """
    if not BITSANDBYTES_AVAILABLE:
        raise RuntimeError("bitsandbytes is required for quantization but not available")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across GPUs
    )

    logger.info("Model loaded with 4-bit quantization")

    return model


def _get_target_modules(model: PreTrainedModel, config: Any) -> list:
    """
    Get appropriate LoRA target modules based on model architecture.

    Args:
        model: The model to apply LoRA to.
        config: Configuration object (may contain target_modules override).

    Returns:
        List of module names to target with LoRA.
    """
    # If explicitly specified in config, use that
    if hasattr(config.model, 'target_modules') and config.model.target_modules:
        logger.info(f"Using target_modules from config: {config.model.target_modules}")
        return config.model.target_modules

    # Auto-detect based on model type
    model_type = model.config.model_type.lower()

    if "llama" in model_type:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "gpt2" in model_type or "gpt-2" in model_type:
        target_modules = ["c_attn"]
    elif "mistral" in model_type:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "phi" in model_type:
        target_modules = ["q_proj", "k_proj", "v_proj", "dense"]
    else:
        # Default fallback - try common attention modules
        logger.warning(f"Unknown model type '{model_type}', using default target modules")
        target_modules = ["q_proj", "v_proj"]

    logger.info(f"Auto-detected target_modules for {model_type}: {target_modules}")
    return target_modules


def _apply_lora(
    model: PreTrainedModel,
    config: Any,
    is_quantized: bool
) -> PeftModel:
    """
    Apply LoRA adapters to the model.

    Args:
        model: Base model to apply LoRA to.
        config: Configuration object with LoRA settings.
        is_quantized: Whether the model is quantized.

    Returns:
        Model with LoRA adapters.
    """
    logger.info("Applying LoRA adapters")

    # Prepare model for k-bit training if quantized
    if is_quantized:
        model = prepare_model_for_kbit_training(model)

    # Get target modules (auto-detect or from config)
    target_modules = _get_target_modules(model, config)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    logger.info(
        f"LoRA applied: rank={config.model.lora_rank}, "
        f"alpha={config.model.lora_alpha}, "
        f"target_modules={target_modules}"
    )

    return model


def prepare_model_for_training(model: PeftModel) -> PeftModel:
    """
    Prepare model for training by freezing base weights and enabling LoRA gradients.

    This function ensures that only LoRA parameters are trainable.

    Args:
        model: Model with LoRA adapters.

    Returns:
        Model ready for training.
    """
    # Freeze all base model parameters
    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Set model to training mode
    model.train()

    # Log trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Training preparation: {trainable:,} / {total:,} parameters trainable")

    return model


def save_model(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    save_path: Union[str, Path],
    save_full_model: bool = False
) -> None:
    """
    Save model and tokenizer to disk.

    For LoRA models, by default only saves the adapter weights (much smaller).
    Set save_full_model=True to save the entire model.

    Args:
        model: Model to save.
        tokenizer: Tokenizer to save.
        save_path: Directory path to save to.
        save_full_model: If True, save full model. If False, save only LoRA adapters.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {save_path}")

    if save_full_model:
        # Save full model
        model.save_pretrained(save_path)
        logger.info("Saved full model with LoRA adapters merged")
    else:
        # Save only LoRA adapters (much smaller)
        if isinstance(model, PeftModel):
            model.save_pretrained(save_path)
            logger.info("Saved LoRA adapters only")
        else:
            logger.warning("Model is not a PeftModel, saving full model")
            model.save_pretrained(save_path)

    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    logger.info("Saved tokenizer")


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    config: Any,
    device: Optional[torch.device] = None
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory.
        config: Configuration object (needed for base model loading).
        device: Target device. If None, auto-detected.

    Returns:
        Tuple of (model, tokenizer).
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if device is None:
        device = get_device()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )

    # Load base model first
    base_model_name = config.model.name
    use_quantization = config.model.use_quantization and is_cuda_available() and BITSANDBYTES_AVAILABLE

    if use_quantization:
        base_model = _load_quantized_model(base_model_name)
    else:
        base_model = _load_standard_model(base_model_name, device)

    # Load LoRA adapters
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    logger.info("Checkpoint loaded successfully")

    return model, tokenizer


def get_model_memory_footprint(model: PreTrainedModel) -> Dict[str, float]:
    """
    Get memory footprint of the model.

    Args:
        model: Model to analyze.

    Returns:
        Dictionary with memory statistics in GB.
    """
    param_memory = sum(p.element_size() * p.numel() for p in model.parameters()) / 1e9
    buffer_memory = sum(b.element_size() * b.numel() for b in model.buffers()) / 1e9
    total_memory = param_memory + buffer_memory

    stats = {
        "parameters_gb": param_memory,
        "buffers_gb": buffer_memory,
        "total_gb": total_memory,
    }

    logger.info(
        f"Model memory footprint: {total_memory:.2f} GB "
        f"(params: {param_memory:.2f} GB, buffers: {buffer_memory:.2f} GB)"
    )

    return stats


def print_model_info(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
    """
    Print detailed information about the model and tokenizer.

    Args:
        model: Model to analyze.
        tokenizer: Tokenizer to analyze.
    """
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)

    # Model architecture
    print(f"Model: {model.config._name_or_path}")
    print(f"Architecture: {model.config.model_type}")

    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    # Memory
    memory = get_model_memory_footprint(model)
    print(f"Memory footprint: {memory['total_gb']:.2f} GB")

    # Tokenizer
    print(f"\nTokenizer vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"EOS token: {tokenizer.eos_token}")

    # Device
    device = next(model.parameters()).device
    print(f"Device: {device}")

    print("=" * 60 + "\n")
