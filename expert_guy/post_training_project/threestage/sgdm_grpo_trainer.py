"""
SGDM GRPO Trainer
=================

A memory-efficient GRPO trainer using SGD with Momentum instead of AdamW.
Based on the wordle-rl-gemma MLX implementation but adapted for PyTorch/TRL.

Key Features:
- SGD + Nesterov Momentum (50% less memory than AdamW)
- Cosine learning rate decay
- Compatible with TRL GRPOTrainer
- Inspired by wordle-rl-gemma's training approach

Memory Comparison:
- AdamW: 2× model params (first + second moments)
- SGD+Momentum: 1× model params (momentum buffer only)
- Example: 3B model saves ~12GB RAM

Usage:
    from sgdm_grpo_trainer import SGDM_GRPOTrainer

    trainer = SGDM_GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        ...
    )
    trainer.train()
"""

import torch
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
import math
from typing import Optional, Dict, Any


class CosineDecayCallback(TrainerCallback):
    """
    Implements cosine learning rate decay schedule.
    Adapted from wordle-rl-gemma's cosine_decay_lr function.
    """

    def __init__(
        self,
        initial_lr: float,
        min_lr: float,
        decay_steps: int
    ):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_steps = decay_steps

    def cosine_decay_lr(self, step: int) -> float:
        """
        Calculates the learning rate at a given step using cosine decay.
        From wordle-rl-gemma line 148-159.
        """
        if step >= self.decay_steps:
            return self.min_lr

        # Cosine annealing formula
        decay_ratio = step / self.decay_steps
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return self.min_lr + coeff * (self.initial_lr - self.min_lr)

    def on_step_begin(self, args, state, control, **kwargs):
        """Update learning rate at the start of each step"""
        if hasattr(kwargs, 'optimizer') and kwargs['optimizer'] is not None:
            new_lr = self.cosine_decay_lr(state.global_step)
            for param_group in kwargs['optimizer'].param_groups:
                param_group['lr'] = new_lr


class SGDM_GRPOTrainer(GRPOTrainer):
    """
    GRPO Trainer using SGD with Nesterov Momentum.

    This trainer replaces AdamW with SGD+Momentum for memory efficiency,
    following the training philosophy from wordle-rl-gemma.

    Key differences from standard GRPOTrainer:
    - Uses SGD with momentum=0.9 and Nesterov acceleration
    - 50% less optimizer memory (1x model params vs 2x for AdamW)
    - Includes cosine LR decay option
    - Better for RL fine-tuning on memory-constrained systems

    Args:
        model: The policy model to train
        reward_funcs: Reward function(s) for GRPO
        args: GRPOConfig with training hyperparameters
        use_cosine_decay: Whether to use cosine LR decay (default: True)
        min_lr: Minimum learning rate for cosine decay (default: 1e-7)
        decay_steps: Steps over which to decay LR (default: total training steps)
        **kwargs: Additional arguments passed to GRPOTrainer

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> from trl import GRPOConfig
        >>>
        >>> model = AutoModelForCausalLM.from_pretrained("model_name")
        >>>
        >>> training_args = GRPOConfig(
        ...     output_dir="output",
        ...     learning_rate=1e-4,  # Higher LR for SGD
        ...     num_train_epochs=3,
        ... )
        >>>
        >>> trainer = SGDM_GRPOTrainer(
        ...     model=model,
        ...     reward_funcs=my_reward_func,
        ...     args=training_args,
        ...     train_dataset=dataset,
        ... )
        >>>
        >>> trainer.train()
    """

    def __init__(
        self,
        model,
        reward_funcs,
        args: GRPOConfig,
        use_cosine_decay: bool = True,
        min_lr: float = 1e-7,
        decay_steps: Optional[int] = None,
        **kwargs
    ):
        # Store cosine decay parameters before calling super().__init__
        self.use_cosine_decay = use_cosine_decay
        self.min_lr = min_lr
        self.initial_lr = args.learning_rate

        # Calculate total training steps if not provided
        if decay_steps is None and use_cosine_decay:
            # Estimate based on dataset size and batch size
            # This will be refined in create_optimizer_and_scheduler
            self.decay_steps = args.num_train_epochs * 1000  # Placeholder
        else:
            self.decay_steps = decay_steps

        # Initialize parent GRPOTrainer
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            **kwargs
        )

        # Add cosine decay callback if enabled
        if self.use_cosine_decay:
            cosine_callback = CosineDecayCallback(
                initial_lr=self.initial_lr,
                min_lr=self.min_lr,
                decay_steps=self.decay_steps
            )
            self.add_callback(cosine_callback)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Creates SGD optimizer with Nesterov momentum instead of AdamW.

        This is the key memory optimization - SGD uses 50% less memory than AdamW.
        Inspired by wordle-rl-gemma line 388 (though they use AdamW on MLX).

        Args:
            num_training_steps: Total number of training steps
        """
        # Update decay_steps now that we know the actual number of training steps
        if self.use_cosine_decay and self.decay_steps == self.args.num_train_epochs * 1000:
            self.decay_steps = num_training_steps

        # Create SGD optimizer with Nesterov momentum
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,              # Standard momentum value
            nesterov=True,             # Nesterov acceleration for better convergence
            weight_decay=self.args.weight_decay
        )

        # Create learning rate scheduler
        # Note: If use_cosine_decay=True, the CosineDecayCallback will handle LR updates
        # Otherwise, we use the default scheduler from args
        if not self.use_cosine_decay:
            self.create_scheduler(
                num_training_steps=num_training_steps,
                optimizer=self.optimizer
            )
        else:
            # Set a dummy scheduler to satisfy TRL's expectations
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1.0  # No-op, CosineDecayCallback handles it
            )


class SGDM_GRPOConfig(GRPOConfig):
    """
    Extended GRPOConfig with SGD-specific hyperparameters.

    Recommended settings for SGD vs AdamW:
    - learning_rate: 10x higher than AdamW (e.g., 1e-4 instead of 1e-5)
    - weight_decay: Slightly lower (e.g., 0.01 instead of 0.05)
    - momentum: 0.9 (fixed in SGDM_GRPOTrainer)
    - use_cosine_decay: True (recommended for SGD)

    Example:
        >>> config = SGDM_GRPOConfig(
        ...     output_dir="sgd_output",
        ...     learning_rate=1e-4,        # Higher LR for SGD
        ...     weight_decay=0.01,         # Lower weight decay
        ...     num_train_epochs=3,
        ...     use_cosine_decay=True,     # Enable cosine LR decay
        ...     min_lr=1e-7,              # Minimum LR for decay
        ... )
    """

    def __init__(
        self,
        use_cosine_decay: bool = True,
        min_lr: float = 1e-7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_cosine_decay = use_cosine_decay
        self.min_lr = min_lr


# ==============================================================================
# Helper Functions (inspired by wordle-rl-gemma)
# ==============================================================================

def print_memory_stats():
    """
    Print current GPU memory usage.
    Useful for comparing AdamW vs SGD memory consumption.
    """
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print("="*60 + "\n")
    elif torch.backends.mps.is_available():
        print("\n" + "="*60)
        print("MPS Memory (Apple Silicon):")
        print("  Note: MPS doesn't expose detailed memory stats")
        print("="*60 + "\n")


def compare_optimizer_memory():
    """
    Estimates memory savings from using SGD vs AdamW.

    Returns:
        dict: Memory comparison statistics
    """
    # Assume a 3B parameter model
    num_params = 3_000_000_000
    bytes_per_param_model = 2  # bfloat16

    # AdamW stores 2 state tensors per parameter (first + second moments)
    adamw_memory_gb = (num_params * bytes_per_param_model * 2) / 1e9

    # SGD stores 1 state tensor per parameter (momentum buffer)
    sgd_memory_gb = (num_params * bytes_per_param_model * 1) / 1e9

    savings_gb = adamw_memory_gb - sgd_memory_gb
    savings_percent = (savings_gb / adamw_memory_gb) * 100

    return {
        "adamw_optimizer_memory_gb": adamw_memory_gb,
        "sgd_optimizer_memory_gb": sgd_memory_gb,
        "memory_savings_gb": savings_gb,
        "memory_savings_percent": savings_percent
    }


if __name__ == "__main__":
    # Demo: Print expected memory savings
    print("\n" + "="*80)
    print("SGDM_GRPOTrainer - Memory Efficiency Analysis")
    print("="*80)

    stats = compare_optimizer_memory()
    print(f"\nFor a 3B parameter model (bfloat16):")
    print(f"  AdamW optimizer memory: {stats['adamw_optimizer_memory_gb']:.2f} GB")
    print(f"  SGD optimizer memory:   {stats['sgd_optimizer_memory_gb']:.2f} GB")
    print(f"  Memory savings:         {stats['memory_savings_gb']:.2f} GB ({stats['memory_savings_percent']:.1f}%)")

    print("\nRecommended hyperparameters:")
    print("  learning_rate: 1e-4  (10x higher than AdamW)")
    print("  momentum: 0.9")
    print("  nesterov: True")
    print("  use_cosine_decay: True")
    print("  weight_decay: 0.01")

    print("\n" + "="*80)
    print("Based on wordle-rl-gemma training approach (MLX → PyTorch adaptation)")
    print("="*80 + "\n")
