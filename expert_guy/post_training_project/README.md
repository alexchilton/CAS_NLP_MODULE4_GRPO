# Wordle LLM Project

## Overview

This project simulates playing the Wordle game using a Hugging Face language model and provides a framework for evaluating model guesses using custom reward functions. It includes logging for all activities and is structured for easy extension and experimentation. The framework supports both Hugging Face datasets and local CSV word lists. It supports RLHF/GRPO training using Hugging Face TRL, with reward functions, distributed/multi-GPU training via accelerate, and PEFT/LoRA fine-tuning.

## Features
- Loads a Hugging Face model and tokenizer using environment variables (`HUGGINGFACE_MODEL_NAME`, `HUGGINGFACE_TOKEN`).
- Defines Wordle game rules and feedback system.
- Uses a prompt template to instruct the model on how to play Wordle.
- Handles model interaction and parses the model's guesses.
- Implements the game loop, feedback mechanism, and win/loss conditions.
- Provides reward functions for evaluating guesses (format, feedback usage, information gain).
- Centralized logging for all activities, with logs saved to `outputs/reward_functions.log`.
- Supports validation and evaluation on both Hugging Face datasets and local CSV word lists.
- Modular, well-documented codebase for easy customization and extension.
- **Supports RLHF/GRPO training with Hugging Face TRL:**
  - Full fine-tuning and PEFT/LoRA training options
  - Integrates custom reward functions and prompt construction
  - Compatible with distributed/multi-GPU training via torchrun/accelerate
  - Supports logging to TensorBoard and Weights & Biases (wandb) for training monitoring

## Requirements
- Python 3.7+
- `transformers` library
- `trl` (Hugging Face TRL)
- `peft` (for LoRA/PEFT training)
- `accelerate` (for distributed/multi-GPU training)
- `python-dotenv` library
- `pandas` library
- `datasets` library (for Hugging Face datasets)
- `scikit-learn` (for data splitting)
- A Hugging Face account and access token

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up a `.env` file in the project directory with the following variables:
   ```env
   HUGGINGFACE_MODEL_NAME=your-model-name
   HUGGINGFACE_TOKEN=your-hf-access-token
   ```

## Usage

### Play a single Wordle game interactively
Run the main script directly:
```bash
python basecase_l3.py
```
By default, the script will play a game of Wordle with the secret word "BRICK".

### Validate model performance on a Hugging Face dataset
```bash
python basecase_l3_dataset.py
```
This script loads a Hugging Face dataset, filters for valid 5-letter words, splits into train/validation, and evaluates model performance on the validation set. Validation statistics are saved to `outputs/validation_stats.json`.

**PEFT version:**
```bash
python basecase_l3_dataset_peft.py
```
Same as above but uses PEFT/LoRA adapters for parameter-efficient fine-tuning.

### Validate model performance on a local CSV word list
```bash
python basecase_l3_local_dataset.py
```
This script loads `five_letter_words.csv`, filters and splits the data, and evaluates model performance on the validation set. Statistics are saved to `outputs/validation_stats.json`.

### RLHF/GRPO Training

#### Full Fine-Tuning
```bash
python grpo_local_data.py
```
Trains the full model with GRPO on the Wordle task. Best for GPUs with sufficient memory.

#### PEFT/LoRA Fine-Tuning (Recommended for Mac/Limited GPU)
```bash
python grpo_local_data_peft.py
```
Trains only LoRA adapters (~1-3% of parameters) with GRPO. Much faster and more memory-efficient while maintaining good performance. **Recommended for Mac or limited GPU memory.**

#### Modular Training Script
```bash
python grpo_local_data_modular.py
```
Modular version with separate reward functions for experimentation.

### Hyperparameter Sweeps

#### Temperature Sweep (Sequential)
```bash
python launch_temperature_sweep.py
```
Runs multiple temperature values sequentially as subprocesses.

#### KL Beta Sweep
```bash
python launch_klbeta_sweep.py
```
Runs multiple KL beta values sequentially to analyze KL regularization effects.

### Distributed/Multi-GPU Training
You can use `accelerate` for distributed training:
```bash
accelerate launch --config_file acc_config.yaml grpo_local_data.py
```
Or use `torchrun` for PyTorch DDP:
```bash
torchrun --nproc_per_node=4 grpo_local_data.py
```
Adjust batch size, gradient accumulation, and `num_generations` as needed in the training scripts.

## Customization
- To change the secret word, modify the argument in the `play_game()` function at the bottom of `basecase_l3.py`.
- To use a different Hugging Face model, update the `HUGGINGFACE_MODEL_NAME` in your `.env` file.
- To adjust reward logic, edit `reward_functions.py` or `reward_functions_base.py`.
- To use a different dataset, update the dataset loading logic in `basecase_l3_dataset.py` or provide a new CSV for `basecase_l3_local_dataset.py`.
- For RLHF/GRPO, edit `grpo_local_data.py` or `grpo_local_data_peft.py` to change model, reward, or training config.

## File Structure

### Core Scripts
- **basecase_l3.py:** Main script for simulating Wordle with a Hugging Face model.
- **basecase_l3_dataset.py:** Validates model performance on a Hugging Face dataset.
- **basecase_l3_dataset_peft.py:** PEFT/LoRA version of dataset validation.
- **basecase_l3_local_dataset.py:** Validates model performance on a local CSV word list.

### GRPO Training Scripts
- **grpo_local_data.py:** Full fine-tuning GRPO training pipeline for Wordle. Uses the entire model.
- **grpo_local_data_peft.py:** PEFT/LoRA version of GRPO training. Only trains LoRA adapters (~1-3% of parameters). **Recommended for Mac/limited GPU memory.**
- **grpo_local_data_modular.py:** Modular GRPO training script with separate reward functions.

### Hyperparameter Sensitivity Scripts
- **grpo_local_data_sensitivity_temperature.py:** Temperature sensitivity analysis (loops over multiple values).
- **grpo_local_data_sensitivity_temperature_subprocess.py:** Single temperature run (accepts `--temperature` argument).
- **grpo_local_data_sensitivity_klbeta_subprocess.py:** Single KL beta run (accepts `--beta` argument).
- **launch_temperature_sweep.py:** Launches sequential temperature sweep via subprocesses.
- **launch_klbeta_sweep.py:** Launches sequential KL beta sweep via subprocesses.

### Utility Scripts
- **reward_functions.py:** Reward functions for evaluating model guesses (format, feedback, information gain).
- **reward_functions_base.py:** Base/modular reward functions.
- **logger_setup.py:** Reusable logger setup, writes to `outputs/reward_functions.log`.
- **plot_loss.py:** Plots and saves training/evaluation loss curves.
- **test_registered_model.py:** Tests registered models.

### Configuration & Data Files
- **five_letter_words.csv:** Example local word list for validation.
- **acc_config.yaml:** Accelerate configuration for distributed training.
- **requirements.txt:** Python dependencies.
- **README.md:** This file.

## Key Differences: Full Fine-Tuning vs PEFT/LoRA

| Feature | Full Fine-Tuning | PEFT/LoRA |
|---------|------------------|-----------|
| **Script** | `grpo_local_data.py` | `grpo_local_data_peft.py` |
| **Parameters Trained** | 100% (all 3B parameters) | ~1-3% (LoRA adapters only) |
| **Speed** | Slower (5-10x per step) | Faster |
| **Memory** | High (~35-45GB for 3B model) | Lower (~15-25GB) |
| **Output Size** | ~6GB (full model) | ~200MB (adapters only) |
| **Performance** | 100% | 95-98% (marginal difference) |
| **Best For** | Large GPU clusters | Mac, single GPU, experimentation |

## Outputs
- **outputs/reward_functions.log:** Centralized log file for all activities and errors.
- **outputs/validation_stats.json:** Validation statistics for model performance.
- **outputs/wordle-grpo/** or **outputs/wordle-grpo-peft/:** GRPO training checkpoints and logs.
- **outputs/** (other): Training logs, checkpoints, and plots.

## Monitoring Training

### Weights & Biases (wandb)
Training metrics are logged to wandb. Key metrics to monitor:

**Primary Metrics:**
- **`train/rewards`** - Should increase over time ✅
- **`eval/loss`** - Should decrease over time ✅

**Secondary Metrics:**
- **`train/kl`** - Should stay relatively low (< 0.5)
- **`train/advantages/mean`** - Positive is good
- **`train/grad_norm`** - Should be stable

**Good training:** Rewards trending up, eval loss trending down, KL staying low.

## Example Output
```
Hugging Face model:
----------------------------------------------------------------------------------------------------
BRICK → Feedback: B(✓) R(✓) I(✓) C(✓) K(✓)
🎉 SUCCESS 🎉
```

## License
This project is provided as-is for educational and research purposes.

## Acknowledgements
This project is inspired by the online course [Reinforcement Fine-Tuning LLMs with GRPO](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/) from DeepLearning.AI.
