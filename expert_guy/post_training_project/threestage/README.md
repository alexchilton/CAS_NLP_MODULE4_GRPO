# Three-Stage Curriculum Learning for Wordle

## Problem Statement

Previous training attempts suffered from **format penalties overwhelming strategy learning**:

- Format errors (invalid words, wrong length) received harsh penalties (-1.5)
- Strategic rewards (feedback usage, info gain) were smaller (+0.4 to +1.0)
- Result: Model struggled to learn strategy because format errors dominated the gradient
- With only 82 SFT examples, format was never properly mastered

## Solution: Curriculum Learning

Separate format learning from strategy learning using a three-stage pipeline:

### Stage 1: Pure Format SFT
**Goal:** Teach output format to 90%+ accuracy

- Generate 1000 synthetic examples focusing on format
- Train with supervised learning (no rewards/penalties)
- Success criteria: 90%+ format accuracy on validation set
- Output: Model that reliably produces `<think>...</think><guess>WORD</guess>`

### Stage 2: Light GRPO with Format Masking
**Goal:** Teach strategic gameplay while maintaining format

- Load Stage 1 model (format already mastered)
- Use **format masking**: minimal format penalties, amplified strategy rewards
- Lower KL penalty to allow exploration
- Success criteria: Improved feedback usage, word accuracy, info gain

**Reward changes in Stage 2:**
```
Format rewards:     +0.8 → +0.3  (minimal, already learned)
Strategy rewards:   +2.0 → +4.2  (amplified 2x)
  - Feedback usage: +1.5 → +3.0
  - Dead letter:    -0.5 → -1.0
  - Info gain:      +1.0 → +1.2
  - Accuracy:       +1.0 → +1.5
```

### Stage 3: Full GRPO Polish
**Goal:** Polish with full constraint enforcement

- Load Stage 2 model (format + basic strategy learned)
- Re-enable all penalties with full strength
- Use staged penalties (lenient early → harsh late)
- Higher KL penalty to prevent drift
- Fewer epochs (polish, not relearn)

## File Structure

```
threestage/
├── README.md                      # This file
├── run_all_stages.py              # Orchestrator script
│
├── generate_sft_data.py           # Generate 1000 synthetic SFT examples
├── stage1_format_sft.py           # Stage 1: Format SFT training
│
├── stage2_reward_functions.py     # Stage 2 rewards (format masking)
├── stage2_light_grpo.py           # Stage 2: Light GRPO training
│
├── stage3_reward_functions.py     # Stage 3 rewards (full system)
├── stage3_full_grpo.py            # Stage 3: Full GRPO training
│
└── [output directories]
    ├── stage1_output/final_model  # Stage 1 trained model
    ├── stage2_output/final_model  # Stage 2 trained model
    └── stage3_output/
        ├── final_model            # Stage 3 final model
        └── best_model             # Best checkpoint (recommended)
```

## Usage

### Quick Start: Run All Stages

```bash
cd threestage
python run_all_stages.py
```

This will:
1. Generate 1000 synthetic SFT examples
2. Train Stage 1 until 90%+ format accuracy
3. Train Stage 2 for strategy learning
4. Train Stage 3 for final polish

### Run Individual Stages

```bash
# Run only Stage 2 (requires Stage 1 complete)
python run_all_stages.py --stage 2

# Force re-run even if already completed
python run_all_stages.py --stage 1 --force

# Skip data generation (use existing data)
python run_all_stages.py --skip-data-gen
```

### Manual Execution

```bash
# Generate data
python generate_sft_data.py

# Stage 1
python stage1_format_sft.py

# Stage 2 (requires Stage 1)
python stage2_light_grpo.py

# Stage 3 (requires Stage 2)
python stage3_full_grpo.py
```

## Expected Results

### Stage 1 Metrics
- Format accuracy: 90%+
- Valid word rate: 90%+
- Training time: ~1-2 hours (1000 examples, 15 epochs)

### Stage 2 Metrics
- Format accuracy: 85%+ (slight drift acceptable)
- Feedback usage improvement: 30-50%
- Dead letter reuse reduction: 40-60%
- Training time: ~2-3 hours (8 epochs)

### Stage 3 Metrics
- Format accuracy: 95%+
- Valid word rate: 95%+
- Feedback compliance: High
- Strategic quality: High
- Training time: ~1-2 hours (5 epochs)

## Key Insights

### Why This Works

1. **Separation of concerns**: Format and strategy are fundamentally different skills
   - Format is syntactic (can be learned from examples)
   - Strategy is semantic (requires reward signals)

2. **Proper reward scaling**: At each stage, rewards match the learning objective
   - Stage 1: No rewards, just supervised learning
   - Stage 2: Strategy rewards >> format rewards
   - Stage 3: Balanced system with staged penalties

3. **Curriculum progression**: Each stage builds on the previous
   - Stage 1 → Stage 2: Format mastered, now learn strategy
   - Stage 2 → Stage 3: Strategy basics learned, now polish

4. **Data efficiency**: 1000 synthetic examples >> 82 real examples
   - Diverse scenarios (first guess, mid-game, late-game)
   - Consistent format (no human annotation errors)
   - Balanced distribution

### Comparison to Previous Approach

| Aspect | Previous (Single Stage) | New (Three Stage) |
|--------|------------------------|-------------------|
| SFT data | 82 examples | 1000 synthetic examples |
| Format learning | Mixed with GRPO | Dedicated SFT stage |
| Reward balance | Format penalties dominate | Stage-appropriate rewards |
| Training time | ~10-20 hours (many rounds) | ~4-7 hours total |
| Success rate | Low (format errors) | High (curriculum) |

## Reward Function Details

### Stage 1: No Rewards
Pure supervised learning - model learns to predict completions given prompts.

### Stage 2: Format Masking
```python
# Format (minimal)
HAS_GUESS_TAG_REWARD = +0.2
NO_GUESS_TAG_PENALTY = -0.5
VALID_WORD_BONUS = +0.1
INVALID_WORD_PENALTY = -0.3

# Strategy (amplified)
CORRECT_POSITION_REWARD = +0.6      (was +0.4)
DEAD_LETTER_PENALTY = -1.0          (was -0.5)
WORD_ACCURACY_WEIGHT = 1.5          (was 1.0)
INFO_GAIN_WEIGHT = 1.2              (was 1.0)
```

### Stage 3: Full System (Staged Penalties)
```python
# Format (full, staged by training progress)
VALID_FORMAT_REWARD = +0.4
INVALID_LENGTH_PENALTY = -1.5 * (0.5 + 0.5 * progress)
INVALID_WORD_PENALTY = -1.5 * (0.5 + 0.5 * progress)

# Strategy (balanced)
CORRECT_POSITION_REWARD = +0.4
DEAD_LETTER_PENALTY = -0.5
WORD_ACCURACY_WEIGHT = 1.0
INFO_GAIN_WEIGHT = 1.0
```

## Troubleshooting

### Stage 1 not reaching 90% accuracy
- Check synthetic data quality: `head sft_synthetic_data.csv`
- Increase epochs: Edit `num_train_epochs` in stage1_format_sft.py
- Increase data: Edit `n` parameters in generate_sft_data.py

### Stage 2 format accuracy drops too much
- Reduce KL penalty: Lower `beta` in stage2_light_grpo.py
- Increase format rewards: Adjust constants in stage2_reward_functions.py
- Reduce epochs: Stop earlier if format degrades

### Stage 3 not improving
- Check if Stage 2 was sufficient: May need more Stage 2 training
- Lower learning rate: Already at 1e-7, but can go lower
- Increase beta: Anchor more strongly to Stage 2

## Next Steps

After training completes:

1. **Evaluate the model:**
   ```bash
   cd ..
   python test_model_comparison.py
   ```

2. **Compare all stages:**
   Create a comparison script to test Stage 1, 2, and 3 models side-by-side

3. **Fine-tune hyperparameters:**
   Adjust learning rates, epochs, or reward weights based on results

4. **Deploy:**
   Use `stage3_output/best_model` for production

## Technical Notes

### Memory Requirements
- Stage 1: ~8-12 GB (SFT with PEFT)
- Stage 2: ~12-16 GB (GRPO with 2 generations)
- Stage 3: ~12-16 GB (GRPO with 2 generations)

### Training Time Estimates (M1 Mac)
- Stage 1: ~1-2 hours
- Stage 2: ~2-3 hours
- Stage 3: ~1-2 hours
- **Total: ~4-7 hours**

Compare to previous single-stage approach: 10-20 hours over multiple rounds with limited success.

### GPU/MPS Optimizations
All scripts include:
- `PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"`
- Gradient accumulation for memory efficiency
- Offloading to disk when needed
- PEFT (LoRA) for parameter efficiency

## References

- Curriculum Learning: Bengio et al. (2009)
- GRPO: Group Relative Policy Optimization
- PEFT/LoRA: Parameter-Efficient Fine-Tuning
- Staged Penalties: Progressive constraint enforcement

## License

MIT (same as parent project)

## Acknowledgments

Built on the Qwen2.5-3B-Instruct base model and TRL's GRPOTrainer.