# Round 5: Where Will It Break?

**Date**: 2025-12-18  
**Status**: Pre-implementation predictions  
**Philosophy**: Assume it will break, predict how, learn from it

---

## The Scientific Mindset

Every previous round broke in unexpected ways:
- Round 1: Weak signals → barely learned
- Round 2: Too harsh → collapsed at -19.95
- Round 3: Reward-focused → format dominated
- Round 4: **We're about to find out what broke!**

Round 5 **will** break. The question is: **How?**

---

## Predicted Failure Modes (Ranked by Likelihood)

### 1. The "Zero-Sum Game" Problem [High Probability]

**Prediction**: Format rewards reduced to 0.8 might be **too low** for initial learning.

**Why this might happen**:
- Current: Format gives 1.5 points → model learns structure quickly
- Round 5: Format gives 0.8 points → might not be enough incentive
- Model could get stuck generating no `<guess>` tags at all
- Without format, strategic rewards can't kick in

**What the logs will show**:
```
Epoch 1: Rewards -2.0 to -1.0 (no format learned)
Epoch 2: Rewards -1.5 to -0.5 (still struggling)
Epoch 3: Rewards -1.0 to 0 (maybe starting to learn)
```

**How we'll know**:
- High rate of completions with no `<guess>` tag (format_reward=0)
- Logs show: "No <guess> tag found at all" repeatedly
- Model doesn't learn basic structure

**Fix if this happens**:
- Round 5.1: Increase VALID_FORMAT_REWARD to 0.6 (midpoint between 0.4 and 1.0)
- Keep penalties the same
- New total: 0.6 + 0.4 = 1.0 (vs current 0.8)

---

### 2. The "Variance Creep" Problem [Medium Probability]

**Prediction**: Variance might creep above 1.5 during actual training.

**Why this might happen**:
- Simulator tested static scenarios
- Actual training has dynamic learning
- As model explores, it might hit extreme cases we didn't test
- INVALID penalties (-1.5) might create occasional -3.0 spikes

**What the logs will show**:
```
Step 50: rewards = [0.5, -2.8, 1.2, -3.1, 0.8]  # High variance
Step 51: rewards = [-0.2, 2.1, -2.5, 0.9, -1.8] # Unstable
```

**How we'll know**:
- Tensorboard shows reward variance increasing over time
- Training loss oscillates instead of decreasing
- Model behavior becomes erratic (good guesses followed by terrible ones)

**Fix if this happens**:
- Round 5.1: Scale all penalties by 0.8x
  - INVALID_LENGTH: -1.5 → -1.2
  - INVALID_WORD: -1.5 → -1.2
  - DEAD_LETTER: -0.7 → -0.56
  - MISSING_GOOD: -0.6 → -0.48
- This reduces variance while maintaining penalty ratios

---

### 3. The "Dead Zone" Problem [Medium Probability]

**Prediction**: Rewards cluster around 0 (neither positive nor negative).

**Why this might happen**:
- Format: +0.8
- Strategic penalties: -0.6 to -0.8
- Net result: +0.0 to +0.2 for most guesses
- Model can't distinguish good from bad

**What the logs will show**:
```
Epoch 3: Mean reward = 0.05 (stuck near zero)
Epoch 5: Mean reward = 0.12 (barely moving)
Epoch 7: Mean reward = 0.08 (no progress)
```

**How we'll know**:
- Reward distribution centered tightly around 0
- Model makes random valid guesses
- No strategic improvement over epochs
- Win rate doesn't increase

**Fix if this happens**:
- Round 5.1: Boost CORRECT_POSITION_REWARD
  - Currently: 0.4 → Increase to 0.8
  - This creates clear positive signal for good play
  - Keeps penalties the same

---

### 4. The "Lowercase Loop" Problem [Low-Medium Probability]

**Prediction**: Model learns that lowercase is slightly rewarded, generates all lowercase.

**Why this might happen**:
- Lowercase gets ~+0.15 (still positive)
- Model discovers this "hack"
- Starts generating: `<guess>storm</guess>` instead of `<guess>STORM</guess>`
- Spreads through training data

**What the logs will show**:
```
Sample 100: <guess>crane</guess> (+0.15)
Sample 150: <guess>storm</guess> (+0.15)
Sample 200: <guess>brick</guess> (+0.15)
# Increasing frequency of lowercase over time
```

**How we'll know**:
- Grep logs for lowercase guesses
- Frequency increases over epochs
- Model "discovers" lowercase gives small positive reward

**Fix if this happens**:
- Round 5.1: Add case checking to `output_format_check()`
  ```python
  if guess != guess.upper():
      reward -= 1.0  # Lowercase penalty
  ```
- This is a **code change**, not just constants

---

### 5. The "Exploration Death" Problem [Low Probability]

**Prediction**: Penalties are too strong, model stops exploring new letters.

**Why this might happen**:
- DEAD_LETTER: -0.7 per letter
- Model tries new letter, gets marked (x), never tries new letters again
- Ends up reusing same safe letters (E, A, R, S, T)
- Info gain collapses

**What the logs will show**:
```
Epoch 1: Avg 4.2 unique letters per guess
Epoch 3: Avg 3.8 unique letters per guess
Epoch 5: Avg 3.2 unique letters per guess
# Decreasing exploration
```

**How we'll know**:
- Info gain rewards decrease over time
- Model guesses become repetitive (always STARE, STORE, etc.)
- Never tries uncommon letters (Q, X, Z)

**Fix if this happens**:
- Round 5.1: Boost EXPLORATION_BONUS
  - Currently: 0.05 → Increase to 0.15
  - Encourages trying new letters
- Or reduce DEAD_LETTER_PENALTY slightly

---

### 6. The "Perfect Score Trap" [Low Probability]

**Prediction**: Model overfits to maximizing reward instead of winning games.

**Why this might happen**:
- High rewards for keeping ✓ positions
- Model learns to make "safe" guesses that preserve checkmarks
- Wins fewer games but gets higher average reward
- Metric gaming instead of strategic play

**What the logs will show**:
```
Epoch 5: Avg reward = +1.8, Win rate = 45%
Epoch 7: Avg reward = +2.1, Win rate = 42%
Epoch 9: Avg reward = +2.4, Win rate = 38%
# Rewards up, wins down!
```

**How we'll know**:
- Reward increases but win rate decreases
- Model makes "conservative" guesses
- Takes 5-6 turns to win when it could win in 3-4

**Fix if this happens**:
- Round 5.1: Add WIN_BONUS = +5.0 for correct final guess
- This aligns reward with actual goal (winning)

---

## The "Unknown Unknown" [Guaranteed]

**Prediction**: Round 5 will break in a way we didn't predict.

**Why this is certain**:
- Every round so far broke in unexpected ways
- We're working with a complex system (LLM + RL + Wordle)
- Emergent behaviors we can't foresee

**What to watch for**:
- Anything that doesn't fit above patterns
- Logs showing behavior that seems "intelligent but wrong"
- Model converging to a local optimum we didn't anticipate

**How to handle**:
1. Don't panic - this is expected!
2. Log the unexpected behavior thoroughly
3. Add it to this document as "Actual Failure #7"
4. Design Round 5.1 to address it

---

## Success Criteria (How to Know If It Works)

Round 5 **succeeds** if it achieves:

### Must Have (Critical)
- [ ] Variance σ < 1.5 throughout all epochs
- [ ] Invalid word rate < 10%
- [ ] Average reward trajectory: -1 → 0 → +1 over epochs
- [ ] Training doesn't collapse (all rewards negative)

### Should Have (Important)
- [ ] Invalid word rate < 5%
- [ ] Win rate > 50% by epoch 10
- [ ] Model uses feedback correctly >50% of time
- [ ] Reward distribution clearly separates good/bad

### Nice to Have (Aspirational)
- [ ] Invalid word rate < 2%
- [ ] Win rate > 70%
- [ ] Variance σ < 1.0
- [ ] Model shows "strategic thinking" in logs

---

## Monitoring Plan for Tomorrow

### First 10 Steps (Critical Window)
Check every step:
- [ ] Are `<guess>` tags appearing?
- [ ] Is variance staying reasonable?
- [ ] Are any rewards positive?

**Red flags**:
- All rewards negative → Format too weak (Failure #1)
- Variance > 2.0 → Penalties too strong (Failure #2)
- No learning visible → Dead zone (Failure #3)

### First 50 Steps (Early Learning)
Check every 10 steps:
- [ ] Reward mean trending upward?
- [ ] Invalid word rate decreasing?
- [ ] Strategic behaviors emerging?

**Red flags**:
- Reward stuck near 0 → Dead zone (Failure #3)
- Lowercase increasing → Loop discovered (Failure #4)
- Exploration decreasing → Over-penalized (Failure #5)

### First Epoch (Full Picture)
At epoch boundary:
- [ ] Compare to baseline (current Round 4)
- [ ] Check variance over full epoch
- [ ] Analyze reward distribution
- [ ] Sample model outputs manually

**Decision point**: Continue or stop and adjust?

---

## The Experimentation Mindset

### What We've Built
1. **Simulator**: Test changes in minutes, not hours
2. **Historical docs**: Learning from each round
3. **Variance analysis**: Mathematical framework for stability
4. **Failure predictions**: Hypothesis-driven debugging

### What Tomorrow Will Teach Us
Round 5 will either:
1. **Work better than expected** → We'll document why and declare victory
2. **Fail as predicted** → We'll apply the pre-planned fix (Round 5.1)
3. **Fail unexpectedly** → We'll document the new failure mode and design Round 5.2

All three outcomes are **valuable learning**.

### The Meta-Learning Loop
```
Round N: Try solution based on Round N-1 failures
         ↓
Observe: Where does it break?
         ↓
Analyze: Why did it break?
         ↓
Predict: What will fix it?
         ↓
Test: Simulate the fix
         ↓
Round N+1: Apply the fix
```

We're on Round 5 of this loop. Each iteration refines our understanding.

---

## Tomorrow's Agenda

**Morning**:
1. Check current training status
2. Apply Round 5 constants to `reward_functions.py`
3. Start Round 5 training
4. Watch first 10 steps like a hawk

**Afternoon**:
1. Check first 50 steps
2. Analyze reward distribution
3. Compare to predictions in this document
4. Decide: continue or adjust?

**Evening**:
1. Check first epoch completion
2. Document what broke (if anything)
3. Update this file with "Actual Failures"
4. Design Round 5.1 fixes if needed

---

## Expected Outcome

**Most likely**: Round 5 will partially work.

It will fix the major issues (invalid words, dead letters) but reveal a new problem (probably format too weak or variance creep).

This is **progress**, not failure!

Each round narrows the problem space. We started with:
- Round 1: Everything broken
- Round 2: Format + variance broken
- Round 3: Variance fixed, strategy broken
- Round 4: Strategy fixed, format domination discovered
- Round 5: Will fix format domination, will discover...?

---

**The real question isn't "Will Round 5 work?"**  
**The real question is: "What will Round 5 teach us about Round 6?"**

Tomorrow we find out! 🔬

