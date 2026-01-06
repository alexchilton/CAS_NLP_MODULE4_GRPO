# Wordle GRPO

Training language models to play Wordle using Group Relative Policy Optimization (GRPO).

## Project Overview

This project implements GRPO (Group Relative Policy Optimization) training for teaching language models to play Wordle effectively. The model learns to make strategic guesses by receiving rewards based on:
- Valid output format
- Use of previous feedback
- Information gain of guesses

## Project Structure

```
wordle-grpo/
├── src/
│   ├── data/              # Data loading and Wordle game logic
│   │   ├── dataset.py     # HuggingFace dataset loader
│   │   └── wordle_game.py # Wordle validation and feedback
│   ├── model/             # Model setup and generation
│   │   ├── setup.py       # Model loading, LoRA, quantization
│   │   └── generation.py  # Memory-efficient text generation
│   ├── training/          # Training logic
│   │   ├── grpo_trainer.py    # GRPO training loop
│   │   └── reward_functions.py # Reward computation
│   ├── evaluation/        # Evaluation utilities
│   └── utils/             # Utility functions
│       ├── device.py      # Device detection (CUDA/MPS/CPU)
│       ├── config.py      # Configuration loading
│       └── logging.py     # Logging and metrics
├── configs/               # Configuration files
│   ├── dev_config.yaml    # Development (Mac, CPU/MPS)
│   └── prod_config.yaml   # Production (GPU, quantization)
├── scripts/               # Training and evaluation scripts
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Evaluation script
│   ├── download_data.py   # Dataset download script
│   ├── view_data.py       # Data viewing utility
│   ├── test_install.py    # Quick installation test
│   └── test_setup.py      # Comprehensive setup verification
├── tests/                 # Unit tests
│   ├── test_wordle_game.py
│   └── test_rewards.py
├── notebooks/             # Jupyter notebooks
│   └── explore_data.ipynb # Data exploration and analysis
└── requirements*.txt      # Dependencies
```

## Setup

### For Development (Mac)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Quick installation test
python scripts/test_install.py

# Download datasets
python scripts/download_data.py

# Comprehensive setup verification
python scripts/test_setup.py
```

### For Production Training (CUDA GPU)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-prod.txt

# Quick installation test
python scripts/test_install.py

# Download datasets
python scripts/download_data.py

# Comprehensive setup verification
python scripts/test_setup.py
```

## Configuration

Two configuration files are provided:

**`configs/dev_config.yaml`** - For development and testing on Mac (CPU/MPS):
- Small model (gpt2)
- No quantization
- Small batch size (1)
- Limited samples (10)

**`configs/prod_config.yaml`** - For production training on GPU:
- Larger model (Qwen2.5-3B-Instruct or Llama 3B)
- 4-bit quantization enabled
- Larger batch size and generations
- Full dataset

Edit these files to customize training parameters, model selection, LoRA settings, etc.

## Usage

### Data Download

Download all required datasets from HuggingFace:

```bash
# Download all datasets (idempotent - safe to run multiple times)
python scripts/download_data.py

# Use custom data directory
python scripts/download_data.py --data-dir /path/to/data

# Force re-download
python scripts/download_data.py --force-download

# Skip verification and samples
python scripts/download_data.py --no-verify --no-samples
```

This downloads:
- `predibase/wordle-grpo` - Main GRPO training dataset
- `predibase/wordle-sft` - Reference SFT dataset
- Word list (extracted from datasets)

All data is cached in `data/` directory for reuse.

### Data Exploration

Explore the datasets interactively with the Jupyter notebook:

```bash
# Install Jupyter (if not already installed)
pip install jupyter matplotlib seaborn

# Launch notebook
jupyter notebook notebooks/explore_data.ipynb
```

The notebook covers:
- Dataset structure and format
- Sample prompts and completions
- Reward function testing on real examples
- Word list statistics and visualizations
- Letter frequency analysis
- What the model needs to learn

### Training

```bash
# Development training (quick test)
python scripts/train.py --config configs/dev_config.yaml

# Production training (full training run)
python scripts/train.py --config configs/prod_config.yaml

# Resume from checkpoint
python scripts/train.py --config configs/prod_config.yaml --resume checkpoints/checkpoint_epoch_5
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py \
    --config configs/prod_config.yaml \
    --checkpoint checkpoints/checkpoint_epoch_10 \
    --num-games 100
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_wordle_game.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Key Components

### Reward Functions

Three reward functions guide learning:

1. **Output Format Check** (`output_format_check`)
   - Validates XML format (`<think>...</think><guess>...</guess>`)
   - Checks word length (5 letters)
   - Verifies word is in valid word list
   - Reward: 0.0 to 1.0

2. **Uses Previous Feedback** (`uses_previous_feedback`)
   - Rewards reusing letters in correct positions (+0.2 each)
   - Rewards trying known letters in new positions (+0.1 each)
   - Penalizes using known wrong letters (-0.5 each)
   - Encourages exploration (+0.05 for new letters)

3. **Guess Value** (`guess_value`)
   - Computes information gain using entropy
   - Rewards guesses that maximize uncertainty reduction
   - Encourages optimal Wordle strategy
   - Reward: 0.0 to 1.0 (normalized)

Combine with `CombinedReward` class to weight multiple objectives.

### GRPO Training

The trainer implements Group Relative Policy Optimization:
- Generates multiple completions per prompt
- Computes rewards for each completion
- Normalizes advantages within groups
- Updates policy using policy gradients
- Includes gradient accumulation and clipping

Training loop (per batch):
1. Generate N completions per prompt
2. Compute rewards → advantages
3. Compute log probabilities
4. Policy gradient loss: -mean(logprob * advantage)
5. Backprop and update LoRA weights

### Memory Management

For 16GB VRAM constraints:
- LoRA fine-tuning (trains 1-5% of parameters)
- 4-bit quantization (production)
- Batch size control
- CUDA cache clearing between batches
- Gradient accumulation

## Model Support

Tested with:
- **gpt2** (small, for dev/testing)
- **Qwen/Qwen2.5-3B-Instruct** (recommended for production)
- **meta-llama/Llama-3.2-3B-Instruct** (alternative)

To use a different model, edit `model.name` in your config file.

## Development

### Code Style

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```

### Adding Features

1. Add implementation in appropriate `src/` module
2. Add tests in `tests/`
3. Update configuration if needed
4. Run tests: `pytest tests/ -v`

### Project Structure Guidelines

- `src/data/` - Data loading, dataset handling, game logic
- `src/model/` - Model loading, generation, setup
- `src/training/` - Training loops, reward functions, optimizers
- `src/evaluation/` - Evaluation metrics and scripts
- `src/utils/` - Shared utilities (device, config, logging)

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` in config
- Reduce `num_generations` in config
- Enable quantization (requires CUDA)
- Increase `gradient_accumulation_steps`

### Slow Training

- Increase `batch_size` if memory allows
- Use GPU with CUDA
- Enable 4-bit quantization
- Reduce `num_generations` per prompt

### bitsandbytes Not Available

- bitsandbytes requires CUDA and is not available on Mac
- Use `use_quantization: false` in config for Mac development
- Quantization automatically disabled if CUDA unavailable

### Model Download Issues

- Check internet connection
- Set `HF_TOKEN` environment variable if model requires authentication
- Use `trust_remote_code: True` for custom models (already set)

## Citation

This project is based on the GRPO algorithm and Wordle training methodology from the BERN AI course.

## License

MIT License
