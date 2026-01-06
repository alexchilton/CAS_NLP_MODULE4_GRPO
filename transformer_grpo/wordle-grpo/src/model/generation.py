"""
Text generation utilities for GRPO training.

This module provides memory-efficient generation functions for creating
multiple completions per prompt, with proper batching and CUDA cache management.
"""

import logging
from typing import Any, List, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from utils.device import is_cuda_available

logger = logging.getLogger(__name__)


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    num_generations: int = 1,
    config: Optional[Any] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 1,
    show_progress: bool = True,
) -> List[List[str]]:
    """
    Generate multiple completions for each prompt with memory-efficient batching.

    This function handles generation for GRPO training, where we need multiple
    samples per prompt. It batches generation to avoid OOM on GPUs with limited VRAM.

    Args:
        model: The language model to generate from.
        tokenizer: Tokenizer for encoding/decoding.
        prompts: List of input prompts.
        num_generations: Number of completions to generate per prompt.
        config: Optional config object containing generation parameters.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling parameter.
        batch_size: Number of prompts to process at once (for memory efficiency).
        show_progress: Whether to show progress bar.

    Returns:
        List of lists, where each inner list contains num_generations completions
        for the corresponding prompt.

    Example:
        >>> prompts = ["Complete this: Hello"]
        >>> completions = generate_completions(model, tokenizer, prompts, num_generations=4)
        >>> len(completions[0])  # 4 completions for first prompt
        4
    """
    # Override with config values if provided
    if config is not None:
        max_new_tokens = getattr(config.model, "max_length", max_new_tokens)
        temperature = getattr(config.training, "temperature", temperature)
        top_p = getattr(config.training, "top_p", top_p)

    model.eval()
    device = next(model.parameters()).device

    all_completions = []
    total_prompts = len(prompts)

    logger.info(
        f"Generating {num_generations} completions for {total_prompts} prompts "
        f"(batch_size={batch_size}, max_new_tokens={max_new_tokens})"
    )

    # Process prompts in batches to avoid OOM
    with torch.no_grad():
        # Create progress bar for prompts
        prompt_iterator = tqdm(
            range(0, total_prompts, batch_size),
            desc="Generating completions",
            disable=not show_progress,
        )

        for batch_start in prompt_iterator:
            batch_end = min(batch_start + batch_size, total_prompts)
            batch_prompts = prompts[batch_start:batch_end]

            # Generate multiple completions for this batch
            batch_completions = _generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )

            all_completions.extend(batch_completions)

            # Clear CUDA cache to free memory
            if is_cuda_available():
                torch.cuda.empty_cache()

            # Update progress bar with memory info
            if is_cuda_available() and show_progress:
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                prompt_iterator.set_postfix({
                    "mem_alloc": f"{memory_allocated:.1f}GB",
                    "mem_resv": f"{memory_reserved:.1f}GB",
                })

    logger.info(f"Generated {len(all_completions)} batches of completions")

    return all_completions


def _generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[List[str]]:
    """
    Generate multiple completions for a batch of prompts.

    To generate num_generations completions per prompt, we repeat each prompt
    num_generations times and generate in one batch for efficiency.

    Args:
        model: The language model.
        tokenizer: Tokenizer.
        prompts: List of prompts for this batch.
        num_generations: Number of completions per prompt.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        device: Device to run on.

    Returns:
        List of lists of completions, one list per input prompt.
    """
    # Repeat each prompt num_generations times
    # e.g., ["prompt1", "prompt2"] with num_generations=3 becomes
    # ["prompt1", "prompt1", "prompt1", "prompt2", "prompt2", "prompt2"]
    repeated_prompts = []
    for prompt in prompts:
        repeated_prompts.extend([prompt] * num_generations)

    # Tokenize all prompts
    inputs = tokenizer(
        repeated_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,  # Reasonable max for input
    ).to(device)

    input_length = inputs.input_ids.shape[1]

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    # Decode outputs
    generated_ids = outputs.sequences
    completions = decode_outputs(
        tokenizer,
        generated_ids,
        skip_special_tokens=True,
        input_length=input_length,
    )

    # Group completions back by original prompt
    # e.g., 6 completions -> [[comp1, comp2, comp3], [comp4, comp5, comp6]]
    grouped_completions = []
    for i in range(len(prompts)):
        start_idx = i * num_generations
        end_idx = start_idx + num_generations
        grouped_completions.append(completions[start_idx:end_idx])

    return grouped_completions


def decode_outputs(
    tokenizer: PreTrainedTokenizer,
    output_ids: torch.Tensor,
    skip_special_tokens: bool = True,
    input_length: Optional[int] = None,
) -> List[str]:
    """
    Decode model outputs to text.

    Args:
        tokenizer: Tokenizer to use for decoding.
        output_ids: Tensor of token IDs (batch_size, sequence_length).
        skip_special_tokens: Whether to skip special tokens in decoded text.
        input_length: If provided, only decode tokens after this position
                     (to get only the generated part, not the prompt).

    Returns:
        List of decoded strings.
    """
    # If input_length provided, slice to get only generated tokens
    if input_length is not None:
        output_ids = output_ids[:, input_length:]

    # Decode
    decoded = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=True,
    )

    # Strip whitespace
    decoded = [text.strip() for text in decoded]

    return decoded


def generate_single(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate a single completion for a single prompt.

    Convenience function for quick testing/inference.

    Args:
        model: The language model.
        tokenizer: Tokenizer.
        prompt: Input prompt.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.

    Returns:
        Generated text completion.
    """
    completions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        num_generations=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=1,
        show_progress=False,
    )

    return completions[0][0]


def estimate_generation_memory(
    batch_size: int,
    num_generations: int,
    max_new_tokens: int,
    model_size_gb: float,
    dtype_bytes: int = 2,
) -> float:
    """
    Estimate GPU memory needed for generation.

    This is a rough estimate to help choose appropriate batch sizes.

    Args:
        batch_size: Number of prompts per batch.
        num_generations: Generations per prompt.
        max_new_tokens: Max tokens to generate.
        model_size_gb: Model size in GB.
        dtype_bytes: Bytes per token (2 for fp16, 4 for fp32).

    Returns:
        Estimated memory in GB.
    """
    # Total sequences being generated
    total_sequences = batch_size * num_generations

    # Rough estimate: each sequence needs storage for activations
    # This is very approximate
    activation_memory_gb = (total_sequences * max_new_tokens * 4096 * dtype_bytes) / 1e9

    # Total estimate
    total_gb = model_size_gb + activation_memory_gb

    logger.info(
        f"Memory estimate: {total_gb:.1f}GB "
        f"(model: {model_size_gb:.1f}GB, activations: {activation_memory_gb:.1f}GB)"
    )

    return total_gb


def get_optimal_batch_size(
    num_prompts: int,
    num_generations: int,
    available_memory_gb: float = 14.0,  # Conservative for 16GB GPU
    model_size_gb: float = 3.0,
) -> int:
    """
    Estimate optimal batch size for generation given memory constraints.

    Args:
        num_prompts: Total number of prompts to generate for.
        num_generations: Generations per prompt.
        available_memory_gb: Available GPU memory in GB.
        model_size_gb: Model size in GB.

    Returns:
        Recommended batch size.
    """
    # Start with batch_size=1 and increase until we exceed memory
    batch_size = 1

    while batch_size <= num_prompts:
        estimated_memory = estimate_generation_memory(
            batch_size=batch_size,
            num_generations=num_generations,
            max_new_tokens=128,
            model_size_gb=model_size_gb,
        )

        if estimated_memory > available_memory_gb:
            # Use previous batch size
            batch_size = max(1, batch_size - 1)
            break

        batch_size += 1

    logger.info(f"Recommended batch_size: {batch_size}")

    return batch_size


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache.

    Call this between generation batches to free up memory.
    """
    if is_cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Log memory stats
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        logger.debug(
            f"GPU memory after clearing: "
            f"allocated={memory_allocated:.2f}GB, reserved={memory_reserved:.2f}GB"
        )


def log_memory_stats() -> None:
    """Log current GPU memory usage."""
    if is_cuda_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        memory_free = (torch.cuda.get_device_properties(0).total_memory / 1e9) - memory_reserved

        logger.info(
            f"GPU memory: allocated={memory_allocated:.2f}GB, "
            f"reserved={memory_reserved:.2f}GB, free={memory_free:.2f}GB"
        )
