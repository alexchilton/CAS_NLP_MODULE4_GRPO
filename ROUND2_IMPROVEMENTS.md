# Round 2 Training Improvements - Symbol Confusion Fix

## Problem Diagnosed

The first model (output2) **completely failed** to understand feedback symbols:
- Treated `O(-)` (letter in word, wrong position) as `O(x)` (letter not in word)
- Treated `R(-)` as dead instead of repositioning it
- Result: Model said "O is not in the word at all" when feedback was `O(-)`

## Root Causes

1. **Weak reward signals**:
   - Using `(-)` correctly: +0.1 (barely rewarding)
   - Ignoring `(-)`: +0.05 (almost same!)
   - Model learned: "Safer to ignore than risk using wrong"

2. **High temperature (1.0)**: Too much chaos, not enough focused learning

3. **No KL regularization (beta=0.0)**: Model drifted into nonsense reasoning

4. **Insufficient training**: Only 5 epochs, model never converged

## Fixes Applied

### 1. Dramatically Strengthened Reward Signals

**Before (Round 1)**:
```python
CORRECT_POSITION_REWARD = 0.2      # Keeping ✓
NEW_POSITION_REWARD = 0.1          # Using (-) at new position
REPEATED_POSITION_PENALTY = -0.2   # Reusing same wrong position
WRONG_LETTER_PENALTY = -1.0        # Dead letter reuse
```

**After (Round 2)**:
```python
CORRECT_POSITION_REWARD = 1.0      # 5x stronger! ✓ positions are CRITICAL
NEW_POSITION_REWARD = 0.5          # 5x stronger! Using (-) correctly is GOOD
REPEATED_POSITION_PENALTY = -1.0   # 5x stronger! Same wrong position is BAD
WRONG_LETTER_PENALTY = -2.0        # 2x stronger! Dead letters HURT
MISSING_VALID_LETTER_PENALTY = -3.0  # NEW! Ignoring (-) letters is CATASTROPHIC
```

### 2. Added Penalty for Ignoring (-) Letters

**New logic** (lines 174-179 in reward_functions.py):
```python
# If feedback shows O(-), model MUST use O in next guess
for letter in valid_letter_to_position.keys():
    if letter not in guess_letters:
        reward += -3.0  # MASSIVE penalty
```

**Example**:
- Feedback: `O(-)` `R(-)`
- Next guess: "STUMP" (no O or R) → **-6.0 penalty** (-3.0 × 2)
- Next guess: "STORM" (has O and R) → **+1.0 reward** (+0.5 × 2)

### 3. Lower Temperature for Focused Learning

**Before**: `temperature=1.0` (high exploration, chaotic)
**After**: `temperature=0.5` (moderate focus, less chaos)

Temperature schedule: 0.5 → 0.3 over first 30% of training

### 4. KL Regularization to Prevent Drift

**Added**: `beta=0.05`

Prevents model from drifting too far from base Qwen model (prevents alphabet hallucinations).

### 5. Gradient Accumulation for Stability

**Before**: `gradient_accumulation_steps=1` (batch size = 2)
**After**: `gradient_accumulation_steps=4` (effective batch = 8)

Smoother, more reliable gradients → better learning.

### 6. More Training Epochs

**Before**: 5 epochs (300 steps)
**After**: 10 epochs (600 steps)

More time to learn the complex symbol distinctions.

## Expected Results

### What Should Improve

1. **Symbol Understanding**:
   - Model will learn `O(-)` means "use O at different position"
   - Model will learn `O(x)` means "never use O again"
   - Model will learn `O(✓)` means "keep O at this exact position"

2. **Reward Trajectory**:
   - Start: -5 to -3 (learning symbols)
   - Middle: -1 to +1 (getting better)
   - End: +1 to +3 (solid understanding)

3. **Behavior**:
   - No more repeated guesses (OGEAR, OGEAR)
   - No more alphabet hallucinations
   - No more treating (-) as (x)

### New Reward Structure Example

**FROST game, after seeing `O(-)` `R(-)`**:

| Model Guess | Reward Breakdown | Total |
|-------------|------------------|-------|
| **STORM** (correct!) | O at new pos: +0.5<br>R at new pos: +0.5<br>S new: +0.05<br>T new: +0.05<br>M new: +0.05 | **+1.15** ✅ |
| **STUMP** (ignores O,R) | Missing O: -3.0<br>Missing R: -3.0<br>S,T,U,M,P: +0.25 | **-5.75** ❌ |
| **OGEAR** (reuses O at same pos) | O at same pos: -1.0<br>R at new pos: +0.5<br>others: +0.15 | **-0.35** ❌ |

## Training Command

```bash
python grpo_local_data_peft.py
```

**Output**: `output3/wordle-grpo-peft/`
**Duration**: ~20-25 hours (4x slower due to gradient accumulation, but much higher quality)

## Files Modified

1. `grpo_local_data_peft.py`:
   - Changed output_dir to output3
   - Increased epochs: 5 → 10
   - Added gradient accumulation: 1 → 4
   - Lowered temperature: 1.0 → 0.5
   - Added KL regularization: beta=0.05
   - Updated temperature callback: 0.5 → 0.3

2. `reward_functions.py`:
   - Increased all rewards 5x
   - Added MISSING_VALID_LETTER_PENALTY = -3.0
   - Added logic to penalize ignoring (-) letters

## Success Criteria

Model passes if it can:
1. ✅ Correctly reposition `O(-)` and `R(-)` letters
2. ✅ Never reuse `(x)` dead letters
3. ✅ Keep `(✓)` letters at correct positions
4. ✅ Average reward > +1.0 by end of training
5. ✅ Win FROST game in ≤6 turns

---

**Date**: 2025-12-18
**Version**: Round 2
**Previous Model**: output2 (failed - symbol confusion)
**New Model**: output3 (in progress)
