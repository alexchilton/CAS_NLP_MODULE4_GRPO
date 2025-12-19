# SFT + GRPO Training Plan (Predibase Approach)

## Why SFT First?

After analyzing the Predibase course materials, we discovered they use a **two-stage approach**:

### Stage 1: SFT (Supervised Fine-Tuning)
- **Purpose**: Teach model the basic format and game rules
- **Dataset**: `predibase/wordle-sft` - examples of good Wordle gameplay
- **Duration**: 10 epochs
- **Output**: Model that understands `<think></think>` and `<guess></guess>` format

### Stage 2: GRPO (starting from SFT checkpoint)
- **Purpose**: Refine strategy using reward functions
- **Dataset**: `predibase/wordle-grpo` - prompts with feedback
- **Duration**: Only 3 epochs (model already knows basics)
- **Output**: Optimized Wordle player

## Why Our Pure GRPO Failed

**Problem**: We tried to do GRPO from scratch (no SFT first)
- Model had to learn format AND strategy simultaneously
- Memory issues from heavy GRPO overhead (generation + gradients)
- Slow convergence (10 epochs barely made progress)

**Predibase's insight**: Separate concerns
1. SFT = Learn the format (simpler, less memory)
2. GRPO = Refine the strategy (focused, fewer epochs)

## Stage 1: SFT Training

### Configuration

```python
# Based on Predibase SFT config
model: "Qwen/Qwen2.5-3B-Instruct"
dataset: "predibase/wordle-sft"
epochs: 10
batch_size: 4  # Larger than GRPO (no generation overhead)
learning_rate: 2e-5  # Higher than GRPO
LoRA rank: 64
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
```

### What SFT Teaches

1. **Format**:
   - Input: Wordle prompts with history
   - Output: `<think>reasoning</think> <guess>WORD</guess>`

2. **Basic Rules**:
   - Guesses must be 5-letter words
   - Use feedback symbols (✓, -, x) correctly
   - Avoid dead letters (x)
   - Reposition wrong-position letters (-)

3. **Valid Words**:
   - Model learns common 5-letter English words
   - Learns to stay within valid word list

### Memory Usage

**SFT is much lighter than GRPO**:
- No generation during training (teacher forcing)
- No KV cache needed
- No reward function overhead
- Standard supervised learning

**Estimated memory**: ~15-20 GB (vs 54 GB for GRPO)

### Training Command

```bash
python sft_training.py
```

**Output**: `output_sft/wordle-sft-peft/final_model/`
**Duration**: ~2-3 hours (much faster than GRPO)

## Stage 2: GRPO Training (from SFT checkpoint)

### Configuration

```python
# Load SFT checkpoint as starting point
base_model: "output_sft/wordle-sft-peft/final_model/"
dataset: "predibase/wordle-grpo"
epochs: 3  # Only 3 epochs (vs 10 for pure GRPO)
batch_size: 2
gradient_accumulation: 4
num_generations: 2
learning_rate: 1e-6  # Lower than SFT
```

### What GRPO Refines

1. **Strategy Optimization**:
   - Which words to guess first (CRANE, ADIEU, etc.)
   - How to maximize information gain
   - When to switch from exploration to exploitation

2. **Feedback Usage**:
   - Optimal repositioning of (-) letters
   - Balancing confirmed (✓) positions with exploration

3. **Edge Cases**:
   - Handling duplicate letters
   - Managing multiple (-) feedback simultaneously

### Reward Functions (Predibase Original)

```python
# Uses exact Predibase values
CORRECT_POSITION_REWARD = 0.2
NEW_POSITION_REWARD = 0.1
REPEATED_POSITION_PENALTY = -0.2
WRONG_LETTER_PENALTY = -0.5
EXPLORATION_REWARD = 0.05
# NO missing letter penalty (trust positive rewards)
```

### Training Command

```bash
# Update grpo_local_data_peft.py to load SFT checkpoint
python grpo_local_data_peft.py
```

**Output**: `output_grpo_from_sft/wordle-grpo-peft/`
**Duration**: ~8-10 hours (3 epochs only)

## Expected Results

### After SFT (Stage 1)
- ✅ Model generates valid format
- ✅ Model uses valid 5-letter words
- ✅ Model understands feedback symbols
- ⚠️ Strategy may be suboptimal

### After GRPO (Stage 2)
- ✅ Optimized word choices
- ✅ Better information gain
- ✅ Correct feedback usage
- ✅ Win rate > 80% (Predibase achieved this)

## Advantages Over Pure GRPO

| Aspect | Pure GRPO | SFT + GRPO |
|--------|-----------|------------|
| Format learning | Slow (10+ epochs) | Fast (10 epochs SFT) |
| Strategy learning | Mixed with format | Focused (3 epochs) |
| Memory usage | 54 GB (failed) | 20 GB SFT + managed GRPO |
| Total time | 30+ hours (failed) | 2-3h SFT + 8-10h GRPO |
| Success rate | Unknown | Proven (Predibase) |

## Implementation Steps

### Step 1: Run SFT Training ✅
```bash
python sft_training.py
```
Wait for completion (~2-3 hours)

### Step 2: Update GRPO Script
Modify `grpo_local_data_peft.py` to:
1. Load SFT checkpoint instead of base model
2. Use Predibase reward values (0.2, 0.1, -0.5)
3. Reduce epochs to 3
4. Keep memory-efficient settings

### Step 3: Run GRPO Training
```bash
python grpo_local_data_peft.py
```
Wait for completion (~8-10 hours)

### Step 4: Test Final Model
```bash
python test_trained_model.py
```

## Success Criteria

**After SFT**:
1. Model generates valid `<think></think>` and `<guess></guess>` format
2. All guesses are valid 5-letter words
3. Model understands feedback symbols

**After GRPO**:
1. Win FROST game in ≤6 turns
2. Average reward > +1.0
3. Correctly repositions (-) letters
4. Avoids (x) dead letters

## Files Created

1. **sft_training.py** - SFT training script (new)
2. **grpo_local_data_peft.py** - Will be updated to load SFT checkpoint
3. **reward_functions.py** - Will revert to Predibase values

---

**Date**: 2025-12-18
**Approach**: Two-stage (SFT → GRPO)
**Based on**: Predibase DeepLearning.AI course
**Previous attempts**:
- Round 1-3: Pure GRPO (failed due to memory/convergence issues)
**New strategy**: Match proven Predibase approach exactly
