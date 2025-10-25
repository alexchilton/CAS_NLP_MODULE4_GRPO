"""
Logging utilities for training and evaluation.

This module provides utilities for setting up loggers, tracking metrics,
and displaying progress during training.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm


def setup_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.

    Args:
        name: Name of the logger (usually __name__).
        log_dir: Directory to save log files. If None, only console logging is used.
        level: Logging level (default: logging.INFO).
        console: Whether to add a console handler (default: True).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Create formatter with timestamp, level, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if log_dir is specified
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsTracker:
    """
    Track and compute statistics for training metrics.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.update("loss", 0.5)
        >>> tracker.update("loss", 0.4)
        >>> tracker.get_average("loss")
        0.45
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.metrics: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}

    def update(self, name: str, value: float, count: int = 1) -> None:
        """
        Update a metric with a new value.

        Args:
            name: Name of the metric.
            value: Value to add.
            count: Number of items this value represents (for weighted averaging).
        """
        if name not in self.metrics:
            self.metrics[name] = []
            self.counts[name] = 0

        self.metrics[name].append(value)
        self.counts[name] += count

    def get_average(self, name: str) -> float:
        """
        Get the average value for a metric.

        Args:
            name: Name of the metric.

        Returns:
            Average value of the metric.

        Raises:
            KeyError: If the metric hasn't been tracked.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' has not been tracked")

        if not self.metrics[name]:
            return 0.0

        return sum(self.metrics[name]) / len(self.metrics[name])

    def get_latest(self, name: str) -> float:
        """
        Get the most recent value for a metric.

        Args:
            name: Name of the metric.

        Returns:
            Most recent value of the metric.

        Raises:
            KeyError: If the metric hasn't been tracked.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' has not been tracked")

        if not self.metrics[name]:
            return 0.0

        return self.metrics[name][-1]

    def get_all(self, name: str) -> List[float]:
        """
        Get all values for a metric.

        Args:
            name: Name of the metric.

        Returns:
            List of all values for the metric.
        """
        return self.metrics.get(name, [])

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset metrics.

        Args:
            name: Name of specific metric to reset. If None, reset all metrics.
        """
        if name is None:
            self.metrics = {}
            self.counts = {}
        elif name in self.metrics:
            self.metrics[name] = []
            self.counts[name] = 0

    def get_summary(self) -> Dict[str, float]:
        """
        Get a summary of all tracked metrics.

        Returns:
            Dictionary mapping metric names to their average values.
        """
        return {name: self.get_average(name) for name in self.metrics.keys()}

    def __repr__(self) -> str:
        """Return string representation of tracked metrics."""
        summary = self.get_summary()
        items = [f"{name}={value:.4f}" for name, value in summary.items()]
        return f"MetricsTracker({', '.join(items)})"


class ProgressBar:
    """
    Simple wrapper around tqdm for consistent progress bar formatting.

    Example:
        >>> pbar = ProgressBar(total=100, desc="Training")
        >>> for i in range(100):
        ...     pbar.update(1)
        ...     pbar.set_postfix({"loss": 0.5})
        >>> pbar.close()
    """

    def __init__(
        self,
        total: Optional[int] = None,
        desc: str = "",
        leave: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize progress bar.

        Args:
            total: Total number of iterations.
            desc: Description to display.
            leave: Whether to leave the progress bar after completion.
            **kwargs: Additional arguments passed to tqdm.
        """
        self.pbar = tqdm(
            total=total,
            desc=desc,
            leave=leave,
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            **kwargs,
        )

    def update(self, n: int = 1) -> None:
        """
        Update progress bar.

        Args:
            n: Number of steps to increment.
        """
        self.pbar.update(n)

    def set_postfix(self, ordered_dict: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Set postfix for progress bar (typically used for metrics).

        Args:
            ordered_dict: Dictionary of values to display.
            **kwargs: Additional key-value pairs to display.
        """
        self.pbar.set_postfix(ordered_dict=ordered_dict, **kwargs)

    def set_description(self, desc: str) -> None:
        """
        Set description for progress bar.

        Args:
            desc: New description.
        """
        self.pbar.set_description(desc)

    def close(self) -> None:
        """Close the progress bar."""
        self.pbar.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        self.close()


def log_metrics(logger: logging.Logger, metrics: Dict[str, Any], prefix: str = "") -> None:
    """
    Log a dictionary of metrics in a formatted way.

    Args:
        logger: Logger instance to use.
        metrics: Dictionary of metric names and values.
        prefix: Optional prefix for the log message.
    """
    metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
    message = f"{prefix} {metrics_str}" if prefix else metrics_str
    logger.info(message)
