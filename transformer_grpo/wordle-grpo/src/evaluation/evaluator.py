"""
Evaluation module for Wordle GRPO models.

This module provides utilities to evaluate trained models by playing
full Wordle games and tracking performance metrics.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from data.wordle_game import validate_guess, is_winning_guess, format_guess_history
from model.generation import generate_single
from training.reward_functions import CombinedReward
from utils.logging import ProgressBar

logger = logging.getLogger(__name__)


class WordleEvaluator:
    """
    Evaluator for Wordle-playing models.

    This class plays full Wordle games with a model and tracks performance metrics
    including win rate, average guesses, and reward scores.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        word_list: List[str],
        max_attempts: int = 6,
        reward_function: Optional[CombinedReward] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model: The model to evaluate.
            tokenizer: Tokenizer for the model.
            word_list: List of valid Wordle words.
            max_attempts: Maximum number of guesses allowed per game.
            reward_function: Optional reward function to compute scores.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.word_list = [w.upper().strip() for w in word_list]
        self.word_set = set(self.word_list)  # Fast O(1) lookup for validation
        self.max_attempts = max_attempts
        self.reward_function = reward_function

        logger.info(
            f"WordleEvaluator initialized with {len(self.word_list)} words, "
            f"max_attempts={max_attempts}"
        )

    def play_game(
        self,
        secret_word: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Play a single Wordle game.

        Args:
            secret_word: The secret word to guess. If None, picks random word.
            verbose: If True, print game progress.

        Returns:
            Dictionary containing game results:
            - secret_word: The secret word
            - guesses: List of guesses made
            - feedbacks: List of feedback strings
            - won: Whether the game was won
            - num_guesses: Number of guesses made
            - rewards: List of rewards for each guess (if reward_function provided)
        """
        # Pick secret word
        if secret_word is None:
            secret_word = random.choice(self.word_list)
        else:
            secret_word = secret_word.upper().strip()

        if verbose:
            logger.info(f"Starting game with secret word: {secret_word}")

        # Game state
        guesses = []
        feedbacks = []
        rewards = []
        won = False

        # Play game
        for attempt in range(self.max_attempts):
            # Create prompt for model
            prompt = self._create_prompt(guesses, feedbacks)

            if verbose:
                logger.info(f"Attempt {attempt + 1}/{self.max_attempts}")
                logger.info(f"Prompt: {prompt[:100]}...")

            # Generate guess
            try:
                completion = generate_single(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                )

                # Extract guess from completion
                guess = self._extract_guess(completion)

                if verbose:
                    logger.info(f"Completion: {completion}")
                    logger.info(f"Extracted guess: {guess}")

            except Exception as e:
                logger.error(f"Error generating guess: {e}")
                guess = None

            # Validate guess
            if guess is None or len(guess) != 5 or guess not in self.word_set:
                if verbose:
                    logger.warning(f"Invalid guess: {guess}")
                # Use random valid word as fallback
                guess = random.choice(self.word_list)
                if verbose:
                    logger.info(f"Using fallback guess: {guess}")

            # Get feedback
            feedback = validate_guess(secret_word, guess)
            guesses.append(guess)
            feedbacks.append(feedback)

            if verbose:
                logger.info(f"Guess: {guess} -> {feedback}")

            # Compute reward if function provided
            if self.reward_function is not None:
                example = {
                    "past_guess_history": list(zip(guesses[:-1], feedbacks[:-1])),
                    "word_list": self.word_list,
                }
                reward = self.reward_function(prompt, completion, example)
                rewards.append(reward)

            # Check if won
            if is_winning_guess(feedback):
                won = True
                if verbose:
                    logger.info(f"Won in {len(guesses)} guesses!")
                break

        if verbose and not won:
            logger.info(f"Lost! Secret word was: {secret_word}")

        return {
            "secret_word": secret_word,
            "guesses": guesses,
            "feedbacks": feedbacks,
            "won": won,
            "num_guesses": len(guesses),
            "rewards": rewards,
        }

    def evaluate(
        self,
        num_games: int = 100,
        secret_words: Optional[List[str]] = None,
        save_transcripts: bool = True,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model by playing multiple games.

        Args:
            num_games: Number of games to play.
            secret_words: Optional list of specific words to use as secrets.
            save_transcripts: Whether to save game transcripts.
            output_dir: Directory to save transcripts.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Starting evaluation with {num_games} games")

        if secret_words is None:
            # Pick random words
            secret_words = random.sample(self.word_list, min(num_games, len(self.word_list)))
        else:
            secret_words = [w.upper().strip() for w in secret_words[:num_games]]

        # Play games
        results = []
        pbar = ProgressBar(total=len(secret_words), desc="Evaluating")

        for secret_word in secret_words:
            result = self.play_game(secret_word=secret_word, verbose=False)
            results.append(result)
            pbar.update(1)

            # Update progress bar with current stats
            wins = sum(1 for r in results if r["won"])
            pbar.set_postfix({"win_rate": f"{wins / len(results):.2%}"})

        pbar.close()

        # Compute metrics
        metrics = self._compute_metrics(results)

        logger.info("Evaluation complete")
        logger.info(f"Win rate: {metrics['win_rate']:.2%}")
        logger.info(f"Average guesses (wins only): {metrics['avg_guesses_when_won']:.2f}")

        # Save transcripts
        if save_transcripts:
            if output_dir is None:
                output_dir = Path("evaluation_results")
            self.save_transcripts(results, metrics, output_dir)

        return metrics

    def _create_prompt(self, guesses: List[str], feedbacks: List[str]) -> str:
        """
        Create prompt for the model given game history.

        Args:
            guesses: List of previous guesses.
            feedbacks: List of feedback for previous guesses.

        Returns:
            Prompt string.
        """
        # Create history string
        history = ""
        if guesses:
            for i, (guess, feedback) in enumerate(zip(guesses, feedbacks)):
                history += f"Guess {i + 1}: {guess} -> {feedback}\n"

        # Create prompt
        prompt = f"""You are playing Wordle. Make a strategic guess to find the secret 5-letter word.

{history if history else "No previous guesses yet."}

Think about what you know and make your next guess.

Format your response as:
<think>Your reasoning here</think>
<guess>WORD</guess>

<think>"""

        return prompt

    def _extract_guess(self, completion: str) -> Optional[str]:
        """
        Extract guess from model completion.

        Args:
            completion: Model's completion text.

        Returns:
            Extracted guess word or None if extraction failed.
        """
        import re

        # Add <think> tag if not present
        if not completion.startswith("<think>"):
            completion = "<think>" + completion

        # Try to extract guess from <guess> tags
        match = re.search(r"<guess>\s*([A-Za-z]+)\s*</guess>", completion, re.IGNORECASE)
        if match:
            guess = match.group(1).upper().strip()
            return guess if len(guess) == 5 else None

        # Fallback: try to find any 5-letter word
        words = re.findall(r'\b[A-Z]{5}\b', completion.upper())
        if words:
            return words[0]

        return None

    def _compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute evaluation metrics from game results.

        Args:
            results: List of game result dictionaries.

        Returns:
            Dictionary of metrics.
        """
        total_games = len(results)
        wins = [r for r in results if r["won"]]
        losses = [r for r in results if not r["won"]]

        metrics = {
            "total_games": total_games,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / total_games if total_games > 0 else 0.0,
        }

        # Guess distribution
        if wins:
            metrics["avg_guesses_when_won"] = sum(r["num_guesses"] for r in wins) / len(wins)
            metrics["min_guesses"] = min(r["num_guesses"] for r in wins)
            metrics["max_guesses"] = max(r["num_guesses"] for r in wins)

            # Distribution by number of guesses
            guess_dist = {}
            for r in wins:
                n = r["num_guesses"]
                guess_dist[n] = guess_dist.get(n, 0) + 1
            metrics["guess_distribution"] = guess_dist
        else:
            metrics["avg_guesses_when_won"] = 0.0
            metrics["min_guesses"] = 0
            metrics["max_guesses"] = 0
            metrics["guess_distribution"] = {}

        # Reward metrics if available
        if results[0]["rewards"]:
            all_rewards = [r for result in results for r in result["rewards"]]
            if all_rewards:
                metrics["avg_reward"] = sum(all_rewards) / len(all_rewards)
                metrics["max_reward"] = max(all_rewards)
                metrics["min_reward"] = min(all_rewards)

        return metrics

    def save_transcripts(
        self,
        results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        output_dir: Path
    ) -> None:
        """
        Save game transcripts and metrics to files.

        Args:
            results: List of game results.
            metrics: Evaluation metrics.
            output_dir: Directory to save to.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save individual game transcripts
        transcripts_file = output_dir / f"transcripts_{timestamp}.json"
        with open(transcripts_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved transcripts to {transcripts_file}")

        # Save metrics
        metrics_file = output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")

        # Save human-readable summary
        summary_file = output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Wordle Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total games: {metrics['total_games']}\n")
            f.write(f"Wins: {metrics['wins']}\n")
            f.write(f"Losses: {metrics['losses']}\n")
            f.write(f"Win rate: {metrics['win_rate']:.2%}\n\n")

            if metrics.get('avg_guesses_when_won', 0) > 0:
                f.write(f"Average guesses (when won): {metrics['avg_guesses_when_won']:.2f}\n")
                f.write(f"Min guesses: {metrics['min_guesses']}\n")
                f.write(f"Max guesses: {metrics['max_guesses']}\n\n")

                f.write("Guess distribution:\n")
                for n in sorted(metrics.get('guess_distribution', {}).keys()):
                    count = metrics['guess_distribution'][n]
                    f.write(f"  {n} guesses: {count} games ({count / metrics['wins']:.1%})\n")

            f.write("\n" + "=" * 60 + "\n")
        logger.info(f"Saved summary to {summary_file}")


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    word_list: List[str],
    num_games: int = 100,
    config: Optional[Any] = None,
    output_dir: Optional[Path] = None,
    reward_function: Optional[CombinedReward] = None,
) -> Dict[str, Any]:
    """
    Convenient function to evaluate a model.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer.
        word_list: List of valid words.
        num_games: Number of games to play.
        config: Optional config object.
        output_dir: Where to save results.
        reward_function: Optional reward function.

    Returns:
        Evaluation metrics.
    """
    max_attempts = 6
    if config is not None and hasattr(config, "evaluation"):
        max_attempts = getattr(config.evaluation, "max_attempts", 6)

    evaluator = WordleEvaluator(
        model=model,
        tokenizer=tokenizer,
        word_list=word_list,
        max_attempts=max_attempts,
        reward_function=reward_function,
    )

    return evaluator.evaluate(
        num_games=num_games,
        save_transcripts=True,
        output_dir=output_dir,
    )


def compare_checkpoints(
    checkpoint_paths: List[Path],
    config: Any,
    word_list: List[str],
    num_games: int = 100,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple model checkpoints.

    Args:
        checkpoint_paths: List of paths to checkpoints.
        config: Configuration object.
        word_list: List of valid words.
        num_games: Number of games per checkpoint.
        output_dir: Where to save comparison results.

    Returns:
        Dictionary mapping checkpoint names to their metrics.
    """
    from model.setup import load_checkpoint

    logger.info(f"Comparing {len(checkpoint_paths)} checkpoints")

    # Use same secret words for fair comparison
    secret_words = random.sample(word_list, min(num_games, len(word_list)))

    all_results = {}

    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")

        # Load model
        model, tokenizer = load_checkpoint(checkpoint_path, config)

        # Evaluate
        evaluator = WordleEvaluator(
            model=model,
            tokenizer=tokenizer,
            word_list=word_list,
            max_attempts=6,
        )

        metrics = evaluator.evaluate(
            num_games=num_games,
            secret_words=secret_words,
            save_transcripts=False,
        )

        checkpoint_name = checkpoint_path.name
        all_results[checkpoint_name] = metrics

        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save comparison
    if output_dir is not None:
        comparison_file = Path(output_dir) / "checkpoint_comparison.json"
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comparison_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved comparison to {comparison_file}")

    return all_results


def generate_report(
    results: Dict[str, Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Generate a markdown report comparing multiple evaluation results.

    Args:
        results: Dictionary mapping names to evaluation metrics.
        output_path: Path to save the markdown report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Header
        f.write("# Wordle Model Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Model | Games | Win Rate | Avg Guesses | Min | Max |\n")
        f.write("|-------|-------|----------|-------------|-----|-----|\n")

        for name, metrics in results.items():
            win_rate = metrics.get("win_rate", 0.0)
            avg_guesses = metrics.get("avg_guesses_when_won", 0.0)
            min_guesses = metrics.get("min_guesses", 0)
            max_guesses = metrics.get("max_guesses", 0)

            f.write(
                f"| {name} | {metrics['total_games']} | "
                f"{win_rate:.1%} | {avg_guesses:.2f} | "
                f"{min_guesses} | {max_guesses} |\n"
            )

        # Detailed results
        f.write("\n## Detailed Results\n\n")

        for name, metrics in results.items():
            f.write(f"### {name}\n\n")
            f.write(f"- **Total games**: {metrics['total_games']}\n")
            f.write(f"- **Wins**: {metrics['wins']}\n")
            f.write(f"- **Losses**: {metrics['losses']}\n")
            f.write(f"- **Win rate**: {metrics.get('win_rate', 0.0):.2%}\n")

            if metrics.get('avg_guesses_when_won', 0) > 0:
                f.write(f"- **Average guesses (wins)**: {metrics['avg_guesses_when_won']:.2f}\n")

                # Guess distribution
                if 'guess_distribution' in metrics:
                    f.write("\n**Guess distribution**:\n\n")
                    for n in sorted(metrics['guess_distribution'].keys()):
                        count = metrics['guess_distribution'][n]
                        pct = count / metrics['wins'] * 100 if metrics['wins'] > 0 else 0
                        f.write(f"- {n} guesses: {count} ({pct:.1f}%)\n")

            # Reward metrics if available
            if 'avg_reward' in metrics:
                f.write(f"\n**Reward metrics**:\n")
                f.write(f"- Average: {metrics['avg_reward']:.3f}\n")
                f.write(f"- Max: {metrics['max_reward']:.3f}\n")
                f.write(f"- Min: {metrics['min_reward']:.3f}\n")

            f.write("\n")

        # Best model
        f.write("## Best Model\n\n")
        best_by_win_rate = max(results.items(), key=lambda x: x[1].get("win_rate", 0.0))
        f.write(f"**By win rate**: {best_by_win_rate[0]} ({best_by_win_rate[1]['win_rate']:.1%})\n\n")

        if any(m.get('avg_guesses_when_won', 0) > 0 for m in results.values()):
            best_by_efficiency = min(
                [(name, m) for name, m in results.items() if m.get('avg_guesses_when_won', 0) > 0],
                key=lambda x: x[1]['avg_guesses_when_won']
            )
            f.write(f"**By efficiency**: {best_by_efficiency[0]} ({best_by_efficiency[1]['avg_guesses_when_won']:.2f} avg guesses)\n")

    logger.info(f"Generated report: {output_path}")
