# Wordle AI Training with GRPO & SFT

A complete pipeline for training language models (Qwen 2.5-3B) to play Wordle using **GRPO (Group Relative Policy Optimization)** and **SFT (Supervised Fine-Tuning)**. This project demonstrates reinforcement learning techniques applied to a concrete game-solving task.

## Overview

This project implements a multi-stage training approach:
1. **Teacher Trajectory Generation** - A heuristic solver generates 50,000 high-quality Wordle games
2. **Supervised Fine-Tuning (SFT)** - The base model learns from expert demonstrations
3. **GRPO Reinforcement Learning** - The model is further refined using reward-based optimization

## Project Structure

```
lara/
├── heuristic_solver.py           # Teacher trajectory generation
├── train_sft optim.py            # SFT training pipeline
├── train_grpo_sft_teached.py     # GRPO training with SFT initialization
├── eval_teach_sft.py             # Evaluation script
├── test_teacher_sft.py           # Quick inference test
├── test_sft_heuristic.py         # Full rollout evaluation
├── run_utils.py                  # Utility functions
├── five_letter_words.csv         # Word list (vocabulary)
├── wordle_teacher_trajs.jsonl    # Generated teacher trajectories
├── wordle_sft_pc.jsonl           # Processed data for SFT training
├── requirements_cluster.txt      # Dependencies
├── 4070/                         # Experimental/development code
│   ├── prompt_templates.py       # Few-shot prompt engineering
│   ├── reward_functions.py       # Advanced reward shaping
│   ├── grpo_local_data_peft_lara.py  # GRPO with PEFT
│   ├── wordle_play_agent.py      # Game agent interface
│   └── logger_setup.py           # Logging configuration
└── version_1/                    # Legacy implementations
```

## Key Components

### 1. Heuristic Solver (`heuristic_solver.py`)
Generates training data using a frequency-based heuristic solver:
- Computes Wordle feedback (green/yellow/gray)
- Selects optimal guesses based on letter frequency and position statistics
- Outputs 50k training examples with full game histories

### 2. SFT Training (`train_sft optim.py`)
Supervised fine-tuning on teacher trajectories:
- Loads Qwen2.5-3B with 4-bit quantization
- Applies LoRA adaptation for efficient training
- Uses completion-only loss (masks prompt tokens)
- Trains for 0.3 epochs

### 3. GRPO Training (`train_grpo_sft_teached.py`)
Reinforcement learning to improve upon SFT:

**Reward Components:**
| Reward | Points | Description |
|--------|--------|-------------|
| Parse + Vocab | +0.6 | Valid `<guess>WORD</guess>` format in dictionary |
| Repeat Penalty | -0.3 | Guessing same word twice |
| Constraint Consistency | +0.6 / -0.2 | Words compatible with past feedback |
| Information Gain | Up to +0.8 | Based on candidate reduction |
| Solve Bonus | +2.0 | Finding the target word |

### 4. Evaluation (`eval_teach_sft.py`)
Comprehensive metrics on 500 test games:
- **Solved Rate** - Percentage of games won
- **Average Turns** - Mean turns needed for solved games
- **Parse Rate** - Valid output format extraction
- **In-Vocab Rate** - Valid dictionary words
- **Constraint Rate** - Guesses compatible with feedback history

## Training Pipeline

```
1. Generate Data        →  python heuristic_solver.py
                            └─> wordle_teacher_trajs.jsonl

2. Preprocess Data      →  python run_utils.py
                            └─> wordle_sft_pc.jsonl

3. Train SFT            →  python "train_sft optim.py"
                            └─> qwen-wordle-sft-teacher-masked/

4. Train GRPO           →  python train_grpo_sft_teached.py
                            └─> qwen-wordle-grpo/

5. Evaluate             →  python eval_teach_sft.py
```

### Data Preprocessing (`run_utils.py`)

After the initial SFT training (`qwen-wordle-sft-teacher`) showed the model wasn't learning properly, a preprocessing step was introduced to transform the raw trajectories into a format better suited for completion-only training.

**Transformation:** `wordle_teacher_trajs.jsonl` → `wordle_sft_pc.jsonl`

```python
# Input format (wordle_teacher_trajs.jsonl):
{
    "system": "You are an expert Wordle solver...",
    "user": "Previous guesses...",
    "label": "AUDIO",
    ...
}

# Output format (wordle_sft_pc.jsonl):
{
    "prompt": "<|im_start|>system\nYou are an expert...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n",
    "completion": "AUDIO<|endoftext|>"
}
```

**Key changes:**
1. Applies the Qwen chat template to properly format system/user messages
2. Separates prompt and completion for masked loss training
3. Adds the EOS token to completions

This preprocessing enabled completion-only loss in SFT, where the model only learns to predict the answer (not the prompt), resulting in the improved `qwen-wordle-sft-teacher-masked` model.

## Model Configuration

| Setting | Value |
|---------|-------|
| Base Model | Qwen2.5-3B-Instruct |
| Quantization | 4-bit (NF4) |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

### Training Hyperparameters

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Learning Rate | 5e-5 | 2e-5 |
| Scheduler | Cosine | Cosine |
| Batch Size | 8 | 1 |
| Grad Accumulation | 4 | 16 |
| Epochs/Steps | 0.3 epochs | 1000 steps |
| Temperature | N/A | 0.8 |
| Num Generations | N/A | 8 |

## Input/Output Formats

**Wordle Feedback:**
```
"C(✓) R(-) A(x) N(x) E(x)"
├─ ✓ = green (correct position)
├─ - = yellow (word contains, wrong position)
└─ x = gray (not in word)
```

**Prompt Format:**
```
System: "You are an expert Wordle solver..."
User: "Previous guesses (word -> feedback):
       CRANE: C(x) R(-) A(x) N(x) E(x)

       Return your next guess.
       Output exactly one line: <guess>ABCDE</guess>"
```

**Model Output:**
```
<guess>AUDIO</guess>
```

## Installation

```bash
pip install -r requirements_cluster.txt
```

### Core Dependencies
- transformers==4.57.3
- peft==0.15.2
- trl==0.26.2
- torch==2.5.1+cu121
- datasets==3.5.0

## Hardware Requirements

- GPU with CUDA support (4GB+ VRAM recommended)
- ~100GB storage for full model artifacts
- bfloat16 training with TF32 acceleration

## Trained Models

| Model | Description |
|-------|-------------|
| `qwen-wordle-sft-teacher/` | Initial SFT attempt using raw trajectories (model didn't learn properly) |
| `qwen-wordle-sft-teacher-masked/` | Improved SFT using preprocessed data with completion-only loss |
| `qwen-wordle-grpo/` | GRPO-optimized model initialized from SFT-masked |

**Lesson Learned:** The initial SFT training failed because the model was trained on the full sequence (prompt + completion) without masking. By preprocessing the data to separate prompts and completions (`run_utils.py`), and using completion-only loss, the model learned to focus on generating valid guesses rather than memorizing prompt patterns.

## Key Insights

This project demonstrates:

1. **Teacher Forcing + RL** - Uses expert heuristic trajectories for SFT warmstart, then refines with RL rewards
2. **Constraint-Aware Rewards** - Rewards not just task success but adherence to game rules
3. **Parameter Efficiency** - PEFT+LoRA keeps model trainable on consumer GPUs
4. **Multi-Stage Training** - SFT → GRPO pipeline leverages both supervised and reinforcement learning
5. **Deterministic Evaluation** - Clear metrics for solve rate, constraint adherence, and token validity

## License

This project is part of the CAS NLP Module 4 coursework.
