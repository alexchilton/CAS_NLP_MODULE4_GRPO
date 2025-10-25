#!/usr/bin/env python3
"""
Quick installation test script.

This script verifies that all required packages are installed correctly
and checks device availability (CUDA, MPS, or CPU).

Usage:
    python scripts/test_install.py
"""

import sys
from pathlib import Path


def test_installation():
    """Test that all required packages are installed."""
    print("=" * 60)
    print("Testing Package Installation")
    print("=" * 60)
    print()

    failed = False

    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch not found: {e}")
        failed = True

    # Test Transformers
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers not found: {e}")
        failed = True

    # Test PEFT
    try:
        import peft
        print(f"✅ PEFT version: {peft.__version__}")
    except ImportError as e:
        print(f"❌ PEFT not found: {e}")
        failed = True

    # Test Datasets
    try:
        import datasets
        print(f"✅ Datasets version: {datasets.__version__}")
    except ImportError as e:
        print(f"❌ Datasets not found: {e}")
        failed = True

    print()
    print("=" * 60)
    print("Checking Device Availability")
    print("=" * 60)
    print()

    # Check device availability
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {device_name}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
        elif torch.backends.mps.is_available():
            print(f"✅ MPS (Apple Silicon GPU) available")
            print(f"   This is perfect for Mac development!")
        else:
            print(f"✅ CPU only (this is fine for development)")
            print(f"   Note: Training will be slower on CPU")
    except Exception as e:
        print(f"⚠️  Could not check device: {e}")

    # Test optional packages
    print()
    print("=" * 60)
    print("Optional Packages")
    print("=" * 60)
    print()

    # Test bitsandbytes (optional, CUDA-only)
    try:
        import bitsandbytes
        print(f"✅ bitsandbytes available (version: {bitsandbytes.__version__})")
        print(f"   Enables 4-bit quantization for GPU training")
    except ImportError:
        print(f"⚠️  bitsandbytes not available (OK for Mac/CPU)")
        print(f"   Only needed for GPU quantization")

    # Test Jupyter (optional)
    try:
        import jupyter
        print(f"✅ Jupyter available")
        print(f"   Run: jupyter notebook notebooks/explore_data.ipynb")
    except ImportError:
        print(f"⚠️  Jupyter not installed (optional)")
        print(f"   Install with: pip install jupyter")

    # Test matplotlib (optional, for notebooks)
    try:
        import matplotlib
        print(f"✅ Matplotlib available (version: {matplotlib.__version__})")
    except ImportError:
        print(f"⚠️  Matplotlib not installed (optional, for visualizations)")
        print(f"   Install with: pip install matplotlib seaborn")

    print()
    print("=" * 60)

    if failed:
        print("❌ Some required packages are missing!")
        print()
        print("To install missing packages, run:")
        print("  pip install -r requirements.txt")
        print()
        print("For development (Mac):")
        print("  pip install -r requirements-dev.txt")
        print()
        print("For production (GPU):")
        print("  pip install -r requirements-prod.txt")
        print("=" * 60)
        return False
    else:
        print()
        print("  ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨")
        print()
        print("  ✅ All required packages installed successfully!")
        print()
        print("  ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨")
        print()
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Download data: python scripts/download_data.py")
        print("  2. Run setup test: python scripts/test_setup.py")
        print("  3. View data: python scripts/view_data.py")
        print("  4. Start training: python scripts/train.py --config configs/dev_config.yaml --dry-run")
        print()
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
