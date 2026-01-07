# Round 3 Training Improvements - Reward-Focused Learning

## Problem with Round 2

Round 2 **completely failed** after 1.6 epochs (25 steps):
- **No improvement**: Rewards stayed around -5 to -8 (started at -4.89, ended at -5.63)
- **High variance**: Rewards swung wildly from -2.94 to -17.52
- **Penalties too harsh**: -3.0 per missing (-) letter created catastrophic negative rewards
- **Example**: Missing 3 (-) letters = -9.0 penalty alone → total reward -19.95

### Why Round 2 Failed

The penalties were **TOO overwhelming**:

```python
# Round 2 (FAILED)
CORRECT_POSITION_REWARD = 1.0
NEW_POSITION_REWARD = 0.5
MISSING_VALID_LETTER_PENALTY = -3.0   # ← TOO HARSH
WRONG_LETTER_PENALTY = -2.0 (escalating)  # ← TOO HARSH
```

**Example from logs**:
- Guess "SCARE" after seeing I(-), N(-), T(-)
- Didn't use I, N, T → -3.0 × 3 = -9.0
- Plus other penalties → total reward: **-19.95**
- Model couldn't find positive gradient signals

## Root Insight from User

**User's key observation**: "i think it should be beneficial to get the tick, minus and cross right not penalised"

This is **exactly right**! The model should be **rewarded heavily for correct behavior**, not just penalized for mistakes.

## Round 3 Solution: Reward-Focused Approach

### Philosophy Change

**Round 2 (penalty-focused)**:
- Small rewards for correct actions (+0.5, +1.0)
- Large penalties for mistakes (-3.0, -2.0 escalating)
- Model learned: "Avoid bad things" (but couldn't find good things)

**Round 3 (reward-focused)**:
- **Large rewards for correct actions** (+2.0, +1.0)
- **Small penalties for mistakes** (-0.5, -0.5)
- Model will learn: "Do good things to maximize reward"

### New Reward Structure

```python
# ROUND 3: Reward-focused
CORRECT_POSITION_REWARD = 2.0    # DOUBLED: Strong positive signal for keeping ✓
NEW_POSITION_REWARD = 1.0        # DOUBLED: Strong positive signal for using (-) at new position
REPEATED_POSITION_PENALTY = -0.5 # Light penalty for repeating wrong position
WRONG_LETTER_PENALTY = -0.5      # Light penalty for dead letters
EXPLORATION_REWARD = 0.1         # DOUBLED: Encourage new letters
MISSING_VALID_LETTER_PENALTY = -0.5  # Light penalty for missing (-) letters
```

**Key ratio**: Positive actions (+2.0, +1.0) are **4x to 2x** more valuable than negative actions (-0.5)

### Example Reward Comparison

**FROST game, after seeing O(-) R(-)**:

| Guess | Round 2 Reward | Round 3 Reward | Explanation |
|-------|----------------|----------------|-------------|
| **STORM** (correct!) | +1.15 | **+2.5** | O at new pos: +1.0<br>R at new pos: +1.0<br>S,T,M new: +0.3 |
| **STUMP** (ignores O,R) | -5.75 | **-0.5** | Missing O: -0.5<br>Missing R: -0.5<br>S,T,U,M,P: +0.5 |
| **OGEAR** (reuses O at same pos) | -0.35 | **+0.1** | O at same pos: -0.5<br>R at new pos: +1.0<br>others: +0.3<br>(slight penalty but R exploration helps) |

**Key improvement**:
- Correct behavior (STORM) gets **+2.5** vs +1.15 → 2.2x stronger signal
- Incorrect behavior (STUMP) gets **-0.5** vs -5.75 → 11x less harsh
- Model can now learn from gradients instead of being overwhelmed by penalties

## Training Configuration

Same as Round 2 (these settings were good):
- **Epochs**: 10 (enough for convergence)
- **Gradient accumulation**: 4 (stable updates, effective batch=8)
- **Temperature**: 0.5 → 0.3 (focused learning)
- **KL regularization**: beta=0.05 (prevents drift)
- **Output**: `output4/wordle-grpo-peft/`

## Expected Results

### Reward Trajectory

Round 3 should show **clear upward trend**:
- **Early** (0-2 epochs): -2 to 0 (learning symbol distinction)
- **Middle** (3-6 epochs): 0 to +1.5 (getting good at using feedback)
- **Late** (7-10 epochs): +1.5 to +3.0 (mastering the game)

### Behavior Improvements

1. ✅ **Strong preference** for keeping ✓ positions (+2.0 reward)
2. ✅ **Strong preference** for trying (-) letters at new positions (+1.0 reward)
3. ✅ **Mild avoidance** of dead letters (-0.5 penalty, manageable)
4. ✅ **Exploration encouraged** for new information (+0.1 per letter)

### Success Criteria

Model passes if:
1. ✅ Average reward > +1.0 by epoch 10
2. ✅ Correctly repositions O(-) and R(-) letters
3. ✅ Mostly avoids reusing (x) dead letters
4. ✅ Keeps (✓) letters at correct positions
5. ✅ Wins FROST game in ≤6 turns

## System Prompt Analysis

The system prompt (from `basecase_l3_dataset_peft.py`) is already **excellent**:

```
### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.
```

Plus explicit examples for each symbol (lines 95-109). **No prompt changes needed**.

## Training Command

```bash
python grpo_local_data_peft.py
```

**Output**: `output4/wordle-grpo-peft/`
**Duration**: ~25-30 hours (150 steps × 100 seconds/step)

## Files Modified

1. **reward_functions.py**:
   - Changed all reward constants to reward-focused values
   - CORRECT_POSITION: 1.0 → 2.0
   - NEW_POSITION: 0.5 → 1.0
   - All penalties: reduced to -0.5
   - Removed escalating penalty for dead letters

2. **grpo_local_data_peft.py**:
   - Changed output_dir: output3 → output4
   - Changed run_name: v2 → v3
   - Updated comments to reflect Round 3

## Why This Should Work

**Psychological/RL principle**:
- Positive reinforcement (+2.0 for correct) is **more effective** than punishment (-3.0 for incorrect)
- Model needs **clear gradient signals** pointing toward good behavior
- Heavy penalties create **noise** and **avoidance behavior** rather than learning

**Mathematical principle**:
- Ratio of reward:penalty = 4:1 (2.0 vs -0.5)
- Model will naturally gravitate toward high-reward actions
- Small penalties still guide away from bad actions without overwhelming signal

**Empirical evidence**:
- Round 1: Weak signals (±0.2) → no learning
- Round 2: Strong penalties (-3.0) → reward collapsed to -19.95
- Round 3: Strong rewards (+2.0) → should see positive rewards by epoch 3

---

**Date**: 2025-12-18
**Version**: Round 3
**Previous Models**:
- output1 (Round 1 - weak signals, partial learning)
- output2 (Round 1 final - symbol confusion)
- output3 (Round 2 - penalties too harsh, training stopped at epoch 1.6)
**New Model**: output4 (Round 3 - reward-focused learning)
