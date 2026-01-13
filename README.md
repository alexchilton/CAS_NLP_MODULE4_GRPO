# GRPO for Wordle: A Scientific Failure Analysis

**Course:** BERN Module 4 - Transformer GRPO  
**Student:** Alex Chilton, Lara Nonis 
**Project:** Training Language Models to Play Wordle using Group Relative Policy Optimization

> 📊 **Training Logs:** See [Weights & Biases dashboard](https://wandb.ai/alexchilton/huggingface) for detailed training metrics and experiment tracking

## 📊 Project Overview

This repository documents an experimental investigation into training language models to play Wordle using GRPO (Group Relative Policy Optimization). Rather than showcasing success, this project provides a **comprehensive analysis of failure modes** encountered when applying RL to structured reasoning tasks.

### Key Findings

We discovered **three distinct failure modes** across different model scales:
1. **Format Failure (GPT-2):** Model too small to follow XML structure
2. **Reward Hacking (Qwen 2.5-3B):** Model optimizes for "safe" guesses (repeating "THINK")
3. **Model Collapse (Gemma 3 4B):** Training degraded performance due to context length mismatches

### Critical Issues Identified

- **Context Length Mismatch:** SFT data (4000+ tokens) vs model capacity (512 tokens)
- **Hyperparameter Sensitivity:** `num_generations=2` caused 75% of training steps to be skipped
- **Temperature Instability:** Gemma 3 4B showed drastically different behavior at 0.7 vs 0.3
- **Prompt Engineering Limitations:** Two different prompting strategies both failed
- **Reward Function Design:** Extensive experimentation with reward weights and formulations (7+ iterations documented in `archive/reward_functions_experiments/`)

---

## 📁 Repository Structure

### Root Directory

```
.
├── README.md                    # This file
├── docs/                        # Documentation and presentations
│   ├── scientific_failure_presentation.md        # Main presentation
│   ├── scientific_failure_presentation_v2.md     # Beamer version
│   ├── scientific_failure_presentation_slides.*  # Generated slides
│   ├── LEARNINGS.md                              # Detailed insights (22KB!)
│   ├── DATA_GUIDE.md                             # Dataset documentation
│   ├── DELIVERABLES_PLAN.md                      # Project planning
│   ├── REWARD_SYSTEM_SUMMARY.md                  # Reward function analysis
│   ├── *_IMPROVEMENTS.md                         # Iteration logs
│   ├── transformer_grpo_README.md                # Original project README
│   └── expert_guy_README.md                      # Advanced experiments README
├── tests/                       # Test files and evaluation scripts (17 files)
│   ├── test_reward*.py          # Reward function unit tests
│   ├── compare_reward_functions.py  # Ablation studies
│   ├── test_gemma3_4b_comparison.py  # Base/SFT/GRPO comparison
│   ├── test_model_comparison.py      # Cross-model evaluation
│   ├── test_trained_model.py         # Checkpoint testing
│   ├── test_edge_cases.py            # Edge case validation
│   ├── test_hallucination.py         # Hallucination detection
│   └── transformer_grpo_test_rewards.py  # Original test suite
├── archive/                     # Archived experimental code
│   └── reward_functions_experiments/  # 7+ reward function iterations
│       ├── reward_functions_original.py
│       ├── reward_functions_improved.py
│       ├── reward_functions_strict.py
│       ├── reward_functions_round4_backup.py
│       ├── reward_functions_round5.py
│       └── reward_functions_OLD_BROKEN.py
├── transformer_grpo/            # Initial GRPO implementation (GPT-2/Qwen)
├── expert_guy/                  # Advanced experiments (Gemma 3 4B)
└── whatever/                    # Scratch/experimental code
```

### `transformer_grpo/wordle-grpo/` - Initial Implementation

> 📖 **Detailed Documentation:** See [docs/transformer_grpo_README.md](docs/transformer_grpo_README.md) for complete setup and usage instructions.

**Purpose:** First attempt at GRPO training for Wordle using GPT-2 and Qwen models.

**Key Components:**
- `src/` - Core GRPO implementation
  - `training/grpo_trainer.py` - Custom GRPO trainer
  - `training/reward_functions.py` - Reward function design
  - `evaluation/evaluator.py` - Game playing and evaluation
  - `model/setup.py` - Model initialization with LoRA
- `scripts/` - Training and evaluation scripts
- `configs/` - Configuration files for different models
- `data/` - Predibase dataset cache
- `evaluation_results/` - Game transcripts and metrics

**Key Files:**
- `README.md` - Original project documentation
- `LEARNINGS.md` - Detailed analysis of training issues (22KB of insights!)
- `DATA_GUIDE.md` - Dataset documentation
- `DELIVERABLES_PLAN.md` - Project planning and structure
- `game_log.txt` - GPT-2 gameplay logs (evidence for Slide 8)

**Models Tested:**
- GPT-2 (117M params) - ❌ Failed at format learning
- Qwen 2.5-1.5B-Instruct - ⚠️ Format learning succeeded, strategic learning failed

**Results:**
- 0% win rate after epoch 1
- 100% XML format compliance
- Model stuck in "THINK" loop (reward hacking)

**Dataset Used:**
- SFT: `predibase/wordle-sft`
- GRPO: `predibase/wordle-grpo`

---

### `expert_guy/post_training_project/` - Advanced Experiments

> 📖 **Detailed Documentation:** See [docs/expert_guy_README.md](docs/expert_guy_README.md) for complete project details and methodology.

**Purpose:** Second experimental track focused on Gemma 3 4B with improved methodology and synthetic data attempts.

**Key Components:**

#### Main Training Scripts
- `basecase_l3.py` - Basic GRPO training with Gemma 3 4B
- `basecase_l3_dataset.py` - Training with HuggingFace datasets
- `basecase_l3_dataset_peft.py` - PEFT/LoRA optimized version
- `basecase_l3_local_dataset.py` - Local dataset experiments
- `grpo_local_data.py` - GRPO with local data generation
- `mlx_grpo_wordle.py` - MLX-optimized GRPO implementation

#### Analysis Scripts
- `test_gemma3_4b_comparison.py` - Compare base/SFT/GRPO models
- `test_model_comparison.py` - Model performance comparison
- `compare_reward_functions.py` - Reward function ablation
- `compare_all_versions.py` - Version comparison analysis
- `analyze_variance.py` - Training variance analysis

#### Testing & Evaluation
- `test_trained_model.py` - Evaluate trained checkpoints
- `test_gemma3_4b_single.py` - Single model testing
- `test_base_model_output.txt` - Base model evaluation results
- `grpo_local_data_sensitivity_temperature.py` - Temperature sensitivity tests

#### Key Directories
- `threestage/` - Three-stage training pipeline (SFT → GRPO stages)
  - `generate_sft_data.py` - Synthetic SFT data generation
  - `sft_synthetic_data.jsonl` - Generated training data (1000+ examples)
  - `stage1_format_sft.py` - Stage 1: Format learning
  - `test_base_model_structured_prompts/` - Structured prompt experiments
    - `prompt_system.py` - Advanced prompting system with state tracking
    - `test_gemma.py` - Gemma testing with structured prompts
    - `test_base_qwen.py` - Qwen baseline tests
    - `gemma_results_temp0.1_games150.txt` - Temperature 0.1 results (2.7% win rate)
  - `outputs/` - Training logs and checkpoints
    - `case.log` - GRPO training logs (contains "infinite loop" evidence)
  - `predibase_wordle_grpo_filtered.csv` - Filtered GRPO dataset
  - `predibase_wordle_grpo_full.csv` - Full GRPO dataset

#### Configuration & Documentation
- `sft_config.yaml` - SFT training configuration (max_seq_length: 512)
- `configs/wordle_grpo.toml` - GRPO configuration
- `README.md` - Project documentation
- `SFT_THEN_GRPO_PLAN.md` - Training strategy documentation
- Multiple `ROUND*_IMPROVEMENTS.md` - Iterative improvement logs

#### Results Files
- `results_gemma3_4b_base_20251222_092336.txt` - Base model: 60% win rate (3/5)
- `results_gemma3_4b_sft_20251222_092336.txt` - SFT model: 40% win rate (2/5)
- `results_gemma3_4b_grpo_20251223_160116.txt` - GRPO model: 20% win rate (1/5)
- `model_comparison_results.json` - Comprehensive comparison data
- `sft_training_output.log` - SFT training logs

**Key Finding:** Training made the model **worse**:
- Base Gemma 3 4B: 60% win rate
- After SFT: 40% win rate (context cutoff broke learning)
- After GRPO: 20% win rate (insufficient training, poor initialization)

**Two Prompting Strategies Tested:**
1. **Simple prompt** (`transformer_grpo/wordle-grpo/src/data/prompt_templates.py`)
   - Few-shot examples
   - Direct format instructions
   - Used for GPT-2 experiments

2. **Structured prompt** (`expert_guy/post_training_project/threestage/test_base_model_structured_prompts/prompt_system.py`)
   - Explicit state tracking (Green/Yellow/Gray letters)
   - "Expert AI" framing
   - Detailed reasoning requirements
   - Used for Gemma 3 4B experiments

**Both approaches failed**, demonstrating that prompt engineering alone cannot overcome fundamental capacity or training issues.

---

## 🎯 Presentation

The main deliverable is the **Scientific Failure Presentation**, documenting what went wrong and why.

### Presentation Files
- `scientific_failure_presentation.md` - Main markdown source
- `scientific_failure_presentation_v2.md` - Version with Beamer `.allowframebreaks` tags
- `scientific_failure_presentation_slides.html` - HTML slides (RevealJS)
- `scientific_failure_presentation_slides.pdf` - PDF export

### Slide Structure
1. **Methodology** (Slides 1-5)
   - XML Strategy and reasoning approach
   - SFT data format (Predibase dataset)
   - GRPO data format
   - Two prompting philosophies
   - System architecture

2. **Critical Failures** (Slides 6-7)
   - Context Length Problem (512 vs 4000+ tokens)
   - Temperature Catastrophe (0.7 vs 0.3)

3. **Model-Specific Failures** (Slides 8-12)
   - GPT-2: Format failure (too small)
   - Qwen 2.5-3B: Reward hacking ("THINK" loop)
   - Gemma 3 4B: Model collapse (infinite loops)

4. **Conclusion** (Slide 13)
   - Prerequisites for GRPO success
   - Lessons learned

### Building Slides

```bash
# Generate HTML slides
pandoc scientific_failure_presentation_v2.md \
  -o scientific_failure_presentation_slides.html \
  -t revealjs -s \
  --css=slides_style.css \
  --slide-level=2 \
  -V theme=moon

# Generate PDF
pandoc scientific_failure_presentation_v2.md \
  -o scientific_failure_presentation_slides.pdf \
  -t beamer
```

---

## 🚀 How to Run

> 💡 **For detailed setup instructions:**
> - Initial implementation: See [docs/transformer_grpo_README.md](docs/transformer_grpo_README.md)
> - Advanced experiments: See [docs/expert_guy_README.md](docs/expert_guy_README.md)

### Prerequisites

```bash
# Clone repository
git clone https://github.com/alexchilton/CAS_NLP_MODULE4_GRPO.git
cd CAS_NLP_MODULE4_GRPO

# Install dependencies
pip install -r transformer_grpo/wordle-grpo/requirements.txt
```

### Training Scripts

#### Option 1: Initial Implementation (transformer_grpo)

**Simple GRPO Training (GPT-2/Qwen):**
```bash
cd transformer_grpo/wordle-grpo

# Development training on Mac (small model, limited data)
python scripts/train.py --config configs/dev_config.yaml

# Production training (Qwen 2.5-1.5B, full dataset)
python scripts/train.py --config configs/qwen_config.yaml

# Resume from checkpoint
python scripts/train.py --config configs/qwen_config.yaml \
  --resume checkpoints_qwen/checkpoint_epoch_1
```

**Supervised Fine-Tuning First:**
```bash
# Train SFT baseline
python scripts/sft_train.py --config configs/qwen_config.yaml

# Then run GRPO
python scripts/train.py --config configs/qwen_config.yaml
```

**Evaluation:**
```bash
# Evaluate trained checkpoint
python scripts/evaluate.py \
  --checkpoint checkpoints_qwen/checkpoint_epoch_1 \
  --num-games 20
```

#### Option 2: Advanced Implementation (expert_guy)

**Three-Stage Training Pipeline (Qwen 2.5-3B):**
```bash
cd expert_guy/post_training_project/threestage

# Stage 1: Format Learning (SFT)
python stage1_format_sft.py
# Output: stage1_output/final_model/

# Stage 2: Light GRPO (KL penalty + basic rewards)
python stage2_light_grpo.py
# Output: stage2_output/

# Stage 3: Full GRPO (complex rewards)
python stage3_full_grpo.py
# Output: stage3_output/
```

**Single-Script Training (Gemma 3 4B):**
```bash
cd expert_guy/post_training_project

# GRPO with Predibase data
python basecase_l3_dataset_peft.py

# GRPO with local data
python grpo_local_data_peft.py

# Temperature sensitivity experiments
python grpo_local_data_sensitivity_temperature.py
```

**Testing Trained Models:**
```bash
# Compare base/SFT/GRPO models
python tests/test_gemma3_4b_comparison.py

# Test single checkpoint
python tests/test_trained_model.py

# Edge case testing
python tests/test_edge_cases.py
```

### Configuration Files

**transformer_grpo:**
- `configs/dev_config.yaml` - Development (GPT-2, 10 samples, CPU/MPS)
- `configs/qwen_config.yaml` - Production (Qwen 1.5B, 100 samples, MPS)
- `configs/prod_config.yaml` - Full production (Qwen 3B, all data, CUDA)

**expert_guy:**
- `expert_guy/post_training_project/configs/wordle_grpo.toml` - GRPO parameters
- `expert_guy/post_training_project/threestage/sft_config.yaml` - SFT configuration (note: max_seq_length=512 issue here!)

### Key Parameters to Adjust

**Critical for Success:**
```python
# In config files or script arguments:
num_generations = 8  # NOT 2! (causes 75% skipped steps)
max_seq_length = 4096  # Match your SFT data length (512 is too short)
batch_size = 4  # Higher if you have GPU memory
learning_rate = 3e-7  # Conservative for GRPO
temperature = 0.3  # For inference (0.7 causes hallucinations)
```

**Reward Weights (curriculum learning):**
```python
# Early training (epochs 0-1): Learn format
format_weight = 1.0
feedback_weight = 0.3
value_weight = 0.3

# Later training (epochs 2+): Learn strategy
format_weight = 0.3
feedback_weight = 1.0
value_weight = 1.0
```

### Viewing Results

**WandB Dashboard:**
Visit [https://wandb.ai/alexchilton/huggingface](https://wandb.ai/alexchilton/huggingface) for:
- Training loss curves
- Reward progression
- Generation examples
- Hyperparameter tracking

**Local Logs:**
```bash
# Training logs
expert_guy/post_training_project/threestage/outputs/case.log

# Evaluation results
transformer_grpo/wordle-grpo/evaluation_results/

# Game transcripts
transformer_grpo/wordle-grpo/evaluation_results/transcripts_*.json
```

---

## 🔬 What We Tried

### Experiment 1: GPT-2 Baseline
**Goal:** Test if small models can learn with GRPO  
**Result:** ❌ Complete failure - couldn't follow XML format  
**Location:** `transformer_grpo/wordle-grpo/game_log.txt`

### Experiment 2: Qwen 2.5-1.5B with Predibase Data
**Goal:** Use instruction-tuned model for better format learning  
**Result:** ⚠️ Format learning succeeded (100%), strategic learning failed (0% win rate)  
**Location:** `transformer_grpo/wordle-grpo/evaluation_results/transcripts_20251026_070527.json`

### Experiment 3: Gemma 3 4B with Predibase SFT+GRPO
**Goal:** Larger model should handle both format and strategy  
**Result:** ❌ Training degraded performance (60% → 40% → 20%)  
**Cause:** Context length mismatch broke SFT initialization  
**Location:** `expert_guy/post_training_project/results_gemma3_4b_*.txt`

### Experiment 4: Synthetic SFT Data
**Goal:** Create shorter training examples to avoid context cutoff  
**Result:** ❌ Data quality insufficient  
**Location:** `expert_guy/post_training_project/threestage/sft_synthetic_data.jsonl`  
**Note:** Abandoned in favor of fixing Predibase data issues

### Experiment 5: Temperature Sensitivity
**Goal:** Find stable generation parameters  
**Result:** ✅ Identified critical instability at temp 0.7  
**Finding:** Model hallucinated extra game states at 0.7, performed better at 0.1-0.3  
**Location:** `expert_guy/post_training_project/threestage/test_base_model_structured_prompts/gemma_results_temp0.1_games150.txt`

### Experiment 6: Reward Function Iterations (7+ versions)
**Goal:** Find optimal balance between format compliance and strategic play  
**Approaches Tried:**
1. **Original** - Equal weights for format/feedback/value
2. **Strict** - Heavy penalties for any mistakes  
3. **Improved** - Graduated rewards with partial credit
4. **Round 4** - Focus on information gain
5. **Round 5** - Curriculum learning approach
6. **Base** - Simplified reward structure
7. **OLD_BROKEN** - Early version with numerical issues

**Key Learnings:**
- Format rewards dominated (55% of total) → model optimized for easy objective
- Graduated rewards helped early training but didn't solve strategic learning
- No reward function could overcome context length or `num_generations` issues
- Reward engineering is iterative but has fundamental limits

**Test Suite:** See `tests/test_reward*.py` for ablation studies  
**Archive:** All versions preserved in `archive/reward_functions_experiments/`  
**Analysis:** See `docs/REWARD_SYSTEM_SUMMARY.md` for detailed comparison

**Current Implementation:**
- `expert_guy/post_training_project/reward_functions.py` - Active version
- `transformer_grpo/wordle-grpo/src/training/reward_functions.py` - Original implementation

---

## 📈 Key Metrics

### Training Efficiency
- `num_generations=2`: **75% of training steps skipped** (identical outputs)
- Recommended: `num_generations=8+` for proper GRPO learning signal
- Training time: 9 hours/epoch on Mac MPS

### Model Performance

| Model | Win Rate | Format Compliance | Notes |
|-------|----------|-------------------|-------|
| GPT-2 | 0% | 0% | Hallucinated format |
| Qwen 2.5-3B (Base) | Unknown | N/A | Not tested |
| Qwen 2.5-3B (SFT+GRPO) | 0% | 100% | "THINK" loop |
| Gemma 3 4B (Base) | 60% (3/5) | Good | Best performance |
| Gemma 3 4B (SFT) | 40% (2/5) | Good | Context cutoff damage |
| Gemma 3 4B (GRPO) | 20% (1/5) | Poor | Training made it worse |

### Temperature Effects (Gemma 3 4B Base)

| Temperature | Win Rate | Behavior |
|-------------|----------|----------|
| 0.1 | 2.7% (4/150) | Stable, some invalid guesses (74%) |
| 0.3 | Good | Coherent gameplay |
| 0.7 | ~0% | Hallucinated extra feedback, excessive reasoning |

---

## 💡 Lessons Learned

### 1. Data Quality Matters
**Problem:** Predibase SFT examples averaged 4000+ tokens, but models configured for 512 tokens max.  
**Impact:** The `<guess>` tag was frequently cut off during training, breaking format learning.  
**Solution:** Either truncate/redesign SFT data OR increase context window to match data.

### 2. Hyperparameters Are Critical
**Problem:** `num_generations=2` caused 75% of training steps to skip (identical outputs).  
**Impact:** Insufficient learning signal for GRPO to work effectively.  
**Solution:** Use `num_generations=8+` as recommended in GRPO literature.

### 3. Format Learning ≠ Strategic Learning
**Finding:** Qwen achieved 100% XML compliance but 0% win rate.  
**Insight:** These are fundamentally different capabilities requiring different optimization pressures.  
**Implication:** Reward weights should use curriculum learning (format → strategy).

### 4. Temperature Is a Hidden Hyperparameter
**Finding:** Gemma 3 4B behavior completely changed between temp 0.3 and 0.7.  
**Insight:** RL-trained policies are more sensitive to sampling parameters than base models.  
**Implication:** Temperature must be carefully tuned during GRPO training AND inference.

### 5. Compute Constraints Are Real
**Issue:** Limited to Mac MPS, 9 hours/epoch, small batch sizes.  
**Impact:** Couldn't run proper ablation studies or hyperparameter sweeps.  
**Lesson:** GRPO requires significant compute for proper experimentation.

---

## 🚀 Future Work

### Immediate Next Steps
1. **Fix context length mismatch:**
   - Truncate Predibase SFT data to 512 tokens
   - OR use models with 4096+ context windows
   - Ensure `<guess>` tag always within context

2. **Increase num_generations to 8:**
   - Should reduce skipped steps from 75% to <30%
   - Provides better learning signal for GRPO

3. **Implement curriculum learning:**
   - Phase 1 (Epochs 0-1): High format weight → learn structure
   - Phase 2 (Epochs 2+): High feedback/value weights → learn strategy
   - Automatic transition based on format compliance metrics

4. **Add explicit repetition penalty:**
   - -1.0 reward for repeating previous guess
   - Forces exploration and prevents local minima

### Long-Term Improvements
1. **Baseline SFT-only training:**
   - Establish ceiling performance with pure supervised learning
   - Determine if GRPO adds value over SFT alone

2. **KL divergence penalty:**
   - Prevent policy from drifting too far from SFT initialization
   - May prevent model collapse issues seen with Gemma

3. **Larger compute experiments:**
   - Test on GPU with larger batch sizes
   - Run for 10+ epochs with proper hyperparameters
   - Comprehensive ablation studies

4. **Alternative RL algorithms:**
   - Compare GRPO vs PPO vs DPO
   - Determine best algorithm for structured reasoning tasks

---

## 📚 References

**Datasets:**
- [predibase/wordle-sft](https://huggingface.co/datasets/predibase/wordle-sft) - SFT training data
- [predibase/wordle-grpo](https://huggingface.co/datasets/predibase/wordle-grpo) - GRPO training data

**Models:**
- GPT-2 (117M parameters)
- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) (4-bit quantized)

**Related Work:**
- DeepSeek GRPO implementation
- HuggingFace TRL library (not used - custom implementation)

---

## 🙏 Acknowledgments

This project documents **honest failure** in applying RL to structured reasoning. The goal was not to achieve state-of-the-art Wordle performance, but to thoroughly understand **why GRPO fails** when prerequisites aren't met.

The failures documented here provide valuable warnings for future research:
- ✅ Don't use `num_generations=2` with GRPO
- ✅ Don't mix 4000-token data with 512-token models
- ✅ Don't assume prompt engineering fixes capacity problems
- ✅ Don't ignore temperature sensitivity in RL training

---

## 📄 License

Educational project for BERN Module 4. Code and documentation provided for learning purposes.

---

## 📧 Contact

**Student:** Alex Chilton  
**Course:** BERN Module 4 - Transformer GRPO  
**Institution:** [University Name]

For questions about the experiments or findings, please open an issue in this repository.
