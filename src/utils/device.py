"""
Device detection and configuration utilities for PyTorch.

This module provides utilities to automatically detect and configure
the optimal device (CUDA/MPS/CPU) and data types for training.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.

    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """
    Check if Apple MPS (Metal Performance Shaders) is available.

    MPS provides GPU acceleration on Apple Silicon (M1/M2/M3) Macs.

    Returns:
        bool: True if MPS is available, False otherwise.
    """
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.

    Priority order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon GPUs)
    3. CPU (fallback)

    Returns:
        torch.device: The optimal device for training.
    """
    if is_cuda_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    elif is_mps_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS (Metal Performance Shaders) device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device (no GPU acceleration available)")

    return device


def get_optimal_dtype(device: torch.device = None) -> torch.dtype:
    """
    Get the optimal data type for the given device.

    CUDA supports mixed precision training with float16 for better performance.
    CPU and MPS work better with float32 for stability.

    Args:
        device: The torch device. If None, will auto-detect using get_device().

    Returns:
        torch.dtype: The recommended data type for the device.
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        dtype = torch.float16
        logger.info("Using torch.float16 (FP16) for CUDA device")
    else:
        dtype = torch.float32
        logger.info(f"Using torch.float32 (FP32) for {device.type} device")

    return dtype


def print_device_info() -> None:
    """
    Print detailed information about the current device setup.

    This is useful for debugging and verifying the training environment.
    """
    device = get_device()
    dtype = get_optimal_dtype(device)

    print("=" * 60)
    print("Device Configuration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Device type: {device.type}")
    print(f"Optimal dtype: {dtype}")

    if device.type == "cuda":
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == "mps":
        print("Running on Apple Silicon with MPS acceleration")

    print(f"PyTorch version: {torch.__version__}")
    print("=" * 60)
