"""
GRPO Trainer for Wordle.

This module implements the Group Relative Policy Optimization (GRPO) trainer
for training language models to play Wordle using reward-based learning.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer

from model.generation import generate_completions, clear_gpu_memory
from utils.logging import MetricsTracker, ProgressBar, log_metrics

logger = logging.getLogger(__name__)


class WordleGRPOTrainer:
    """
    GRPO Trainer for Wordle tasks.

    This trainer implements Group Relative Policy Optimization, a variant of
    policy gradient methods that normalizes advantages within groups of samples.

    The training loop:
    1. Generate N completions per prompt
    2. Compute rewards for each completion
    3. Calculate advantages (reward - baseline)
    4. Compute policy gradient loss
    5. Backpropagate and update model
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_function: Callable,
        config: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        save_dir: Optional[Path] = None,
    ):
        """
        Initialize the GRPO trainer.

        Args:
            model: The language model to train (with LoRA adapters).
            tokenizer: Tokenizer for the model.
            reward_function: Function to compute rewards for completions.
            config: Configuration object with training parameters.
            optimizer: Optional optimizer (if None, creates AdamW).
            save_dir: Directory to save checkpoints.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_function = reward_function
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path(config.output.checkpoint_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        self.num_generations = config.training.num_generations
        self.learning_rate = config.training.learning_rate
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps

        # Initialize optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
            )
        else:
            self.optimizer = optimizer

        # Metrics tracking
        self.metrics_tracker = MetricsTracker()
        self.global_step = 0
        self.current_epoch = 0

        # Device
        self.device = next(model.parameters()).device

        logger.info(
            f"GRPO Trainer initialized: "
            f"num_generations={self.num_generations}, "
            f"lr={self.learning_rate}, "
            f"grad_accum={self.gradient_accumulation_steps}"
        )

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader providing training batches.
            epoch: Current epoch number.

        Returns:
            Dictionary of epoch-level metrics.
        """
        self.model.train()
        self.current_epoch = epoch
        self.metrics_tracker.reset()

        logger.info(f"Starting epoch {epoch}")

        # Progress bar for batches
        pbar = ProgressBar(total=len(dataloader), desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(dataloader):
            # Training step
            step_metrics = self._training_step(batch, batch_idx)

            # Update metrics
            for key, value in step_metrics.items():
                self.metrics_tracker.update(key, value)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{step_metrics.get('loss', 0.0):.4f}",
                "reward": f"{step_metrics.get('mean_reward', 0.0):.4f}",
            })

            # Log training step
            if self.global_step % 10 == 0:
                self.log_training_step(self.global_step, step_metrics)

            # Clear GPU memory periodically
            if batch_idx % 5 == 0:
                clear_gpu_memory()

        pbar.close()

        # Epoch summary
        epoch_metrics = self.metrics_tracker.get_summary()
        logger.info(f"Epoch {epoch} completed")
        log_metrics(logger, epoch_metrics, prefix=f"Epoch {epoch}")

        return epoch_metrics

    def _training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Batch of data from dataloader.
            batch_idx: Index of current batch.

        Returns:
            Dictionary of step metrics.
        """
        prompts = batch["prompts"]
        batch_size = len(prompts)

        # Step 1: Generate N completions per prompt
        logger.debug(f"Step {self.global_step}: Generating {self.num_generations} completions")
        all_completions = generate_completions(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            num_generations=self.num_generations,
            config=self.config,
            batch_size=1,  # Generate one prompt at a time for memory efficiency
            show_progress=False,
        )
        # all_completions: List[List[str]], shape (batch_size, num_generations)

        # Step 2: Compute rewards for all completions
        logger.debug(f"Step {self.global_step}: Computing rewards")
        rewards = self._compute_rewards(prompts, all_completions, batch)
        # rewards: torch.Tensor, shape (batch_size, num_generations)

        # Step 3: Compute advantages (group-relative normalization)
        logger.debug(f"Step {self.global_step}: Computing advantages")
        advantages = self._compute_advantages(rewards)
        # advantages: torch.Tensor, shape (batch_size, num_generations)

        # Step 4: Compute log probabilities for generated completions
        logger.debug(f"Step {self.global_step}: Computing log probabilities")
        logprobs = self._compute_log_probabilities(prompts, all_completions)
        # logprobs: torch.Tensor, shape (batch_size, num_generations)

        # Step 5: Compute GRPO loss
        loss = self.compute_grpo_loss(logprobs, advantages)

        # Step 6: Backpropagation with gradient accumulation
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        # Update weights if accumulated enough gradients
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Collect metrics
        metrics = {
            "loss": loss.item() * self.gradient_accumulation_steps,
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "mean_advantage": advantages.mean().item(),
        }

        self.global_step += 1

        return metrics

    def _compute_rewards(
        self,
        prompts: List[str],
        completions: List[List[str]],
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute rewards for all completions.

        Args:
            prompts: List of prompts.
            completions: List of lists of completions (batch_size, num_generations).
            batch: Original batch data (contains example info for reward functions).

        Returns:
            Tensor of rewards, shape (batch_size, num_generations).
        """
        batch_size = len(prompts)
        rewards = torch.zeros(batch_size, self.num_generations)

        for i, prompt in enumerate(prompts):
            for j, completion in enumerate(completions[i]):
                # Create example dict for reward function
                # TODO: Extract proper example data from batch
                example = {
                    "past_guess_history": batch.get("past_guess_histories", [[]])[i],
                    "word_list": batch.get("word_list", []),
                }

                # Compute reward
                reward = self.reward_function(prompt, completion, example)
                rewards[i, j] = reward

        return rewards.to(self.device)

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages using group-relative normalization.

        In GRPO, advantages are computed by normalizing within each group
        (group = all generations for a single prompt).

        Args:
            rewards: Tensor of rewards, shape (batch_size, num_generations).

        Returns:
            Tensor of advantages, shape (batch_size, num_generations).
        """
        # TODO: Implement proper GRPO advantage computation
        # For now, use simple baseline subtraction (mean reward per prompt)

        # Compute baseline (mean reward for each prompt)
        baseline = rewards.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1)

        # Advantages = rewards - baseline
        advantages = rewards - baseline  # Shape: (batch_size, num_generations)

        # Optional: Normalize advantages within each group
        # TODO: Add group-wise normalization if needed
        # advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (advantages.std(dim=1, keepdim=True) + 1e-8)

        return advantages

    def _compute_log_probabilities(
        self,
        prompts: List[str],
        completions: List[List[str]]
    ) -> torch.Tensor:
        """
        Compute log probabilities for generated completions.

        This computes the log probability of each completion under the current policy.

        Args:
            prompts: List of prompts.
            completions: List of lists of completions.

        Returns:
            Tensor of log probabilities, shape (batch_size, num_generations).
        """
        batch_size = len(prompts)
        logprobs = torch.zeros(batch_size, self.num_generations)

        # TODO: Implement efficient batched log probability computation
        # For now, compute one at a time

        for i, prompt in enumerate(prompts):
            for j, completion in enumerate(completions[i]):
                # Combine prompt and completion
                full_text = prompt + completion

                # Tokenize
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)

                prompt_inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)

                prompt_length = prompt_inputs.input_ids.shape[1]

                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                # Compute log probabilities for completion tokens only
                # TODO: Optimize this - currently inefficient
                completion_logprobs = []
                for k in range(prompt_length, inputs.input_ids.shape[1] - 1):
                    token_id = inputs.input_ids[0, k + 1]
                    token_logits = logits[0, k, :]
                    token_logprob = F.log_softmax(token_logits, dim=-1)[token_id]
                    completion_logprobs.append(token_logprob.item())

                # Sum log probabilities
                if completion_logprobs:
                    logprobs[i, j] = sum(completion_logprobs)
                else:
                    logprobs[i, j] = 0.0

        return logprobs.to(self.device)

    def compute_grpo_loss(
        self,
        logprobs: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GRPO policy gradient loss.

        The basic policy gradient loss is:
        L = -mean(logprob * advantage)

        GRPO adds additional terms:
        - KL divergence penalty (to prevent drift from reference model)
        - Group-based normalization

        Args:
            logprobs: Log probabilities, shape (batch_size, num_generations).
            advantages: Advantages, shape (batch_size, num_generations).

        Returns:
            Scalar loss tensor.
        """
        # TODO: Implement full GRPO loss with KL divergence penalty

        # Basic policy gradient loss
        policy_loss = -(logprobs * advantages).mean()

        # TODO: Add KL divergence penalty
        # kl_loss = compute_kl_divergence(current_model, reference_model)
        # total_loss = policy_loss + beta * kl_loss

        # For now, return simple policy gradient loss
        return policy_loss

    def log_training_step(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Log metrics for a training step.

        Args:
            step: Current global step.
            metrics: Dictionary of metrics to log.
        """
        log_metrics(logger, metrics, prefix=f"Step {step}")

        # TODO: Add wandb/tensorboard logging if configured
        # if self.config.logging.wandb.enabled:
        #     wandb.log(metrics, step=step)

    def save_checkpoint(self, epoch: int, metrics: Optional[Dict[str, float]] = None) -> Path:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number.
            metrics: Optional metrics to save with checkpoint.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        from model.setup import save_model
        save_model(self.model, self.tokenizer, checkpoint_path)

        # Save training state
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        torch.save(state, checkpoint_path / "training_state.pt")

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load model checkpoint and training state.

        Args:
            checkpoint_path: Path to checkpoint directory.
        """
        # Load training state
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path)
            self.current_epoch = state["epoch"]
            self.global_step = state["global_step"]
            self.optimizer.load_state_dict(state["optimizer_state"])

            logger.info(
                f"Loaded checkpoint from epoch {self.current_epoch}, "
                f"step {self.global_step}"
            )
        else:
            logger.warning(f"Training state not found at {state_path}")
