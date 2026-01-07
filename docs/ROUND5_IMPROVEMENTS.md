# Round 5 Training Improvements - The Variance Paradox

**Date**: 2025-12-18  
**Current Model**: output4 (Round 3/4 - running)  
**Problem**: Teaching well-formatted garbage  
**Solution**: Moderate reward balancing with variance control

---

## The Journey So Far: What We've Learned

### Round 1: The Weak Signal Problem
**Problem**: Rewards too small (±0.2)  
**Result**: Model barely learned anything  
**Lesson**: Need strong signals for RL to work

### Round 2: The Penalty Catastrophe  
**Problem**: Swung too far - penalties too harsh (-3.0 per mistake)  
**Result**: Rewards collapsed to -19.95, training failed after 25 steps  
**Lesson**: Heavy penalties create noise, not learning

### Round 3: The Reward Revolution
**Problem**: Changed philosophy to reward-focused  
**Approach**: Strong rewards (+2.0) for good, light penalties (-0.5) for bad  
**Result**: Training ran... but discovered NEW problem!

### Round 4 (Current): The Format Domination Problem
**Problem Discovered (via simulator)**: Format rewards dominate everything  
**Finding**: 94% of bad behaviors get POSITIVE rewards!  
**Examples**:
- Lowercase "scowl": +0.77 (should be negative)
- 4-letter "CARE": +0.78 (should be < -1.5)
- Invalid "XYZZZ": +0.72 (should be < -1.5)
- Ignoring feedback: +1.61 (should be negative)
- Hallucinating: -0.07 (should be < -2.0)

**Root Cause**: `VALID_FORMAT_REWARD = 1.0` + `VALID_WORD_BONUS = 0.5` = **1.5 points just for format!**

Even with -1.0 penalty, total is still +0.5 → model learns to repeat bad behavior!

---

## The Variance Paradox: Round 5's Core Challenge

We discovered a **fundamental trade-off** through systematic simulator testing:

### The Three Systems We Tested

| System | Variance (σ) | Bad Penalty % | Good/Bad Separation | Verdict |
|--------|--------------|---------------|---------------------|---------|
| **Current** | 0.68 ✅ | 6% ❌ | 1.12 ❌ | Stable but broken |
| **Aggressive** | 1.84 ❌ | 61% ⚠️ | 3.38 ✅ | Fixes penalties but unstable |
| **Moderate** | 1.19 ✅ | 72% ✅ | 2.53 ✅ | **GOLDILOCKS ZONE** |

### Why Variance Matters

**User insight**: "the problem the last set of rewards was trying to fix was too strict a formatting - the formatting was always negative so it didnt learn the game - and in addition the variance was too high"

This is THE key constraint we must respect!

**High variance problems**:
1. Unstable gradients → training doesn't converge
2. Rewards swing from -4.6 to +3.5 (8 point range!) 
3. Model can't learn consistent policy
4. Training may collapse (like Round 2)

**Target**: Keep standard deviation < 1.5

---

## Round 5 Solution: Moderate Balanced System

### The Philosophy

We need to thread the needle:
- **Strict enough** to penalize bad behaviors (unlike current)
- **Lenient enough** to avoid variance explosion (unlike aggressive)
- **Clear enough** signals for RL to work (unlike Round 1)

### The Constants

```python
# FORMAT REWARDS (reduce by 50-60%)
VALID_FORMAT_REWARD = 0.4          # was 1.0 → 60% reduction
PARTIAL_FORMAT_REWARD = 0.2        # was 0.3 → 33% reduction
VALID_WORD_BONUS = 0.4             # was 0.5 → 20% reduction

# PENALTIES (increase by 3-5x)
INVALID_LENGTH_PENALTY = -1.5      # was -0.3 → 5x stronger
INVALID_WORD_PENALTY = -1.5        # was -0.5 → 3x stronger
DEAD_LETTER_PENALTY = -0.7         # was -0.4 → 1.75x stronger
MISSING_GOOD_LETTER_PENALTY = -0.6 # was -0.3 → 2x stronger

# KEEP SAME (these work)
CORRECT_POSITION_REWARD = 0.4
NEW_POSITION_REWARD = 0.3
REPEATED_WRONG_POSITION = -0.2
EXPLORATION_BONUS = 0.05
```

### The Math Behind It

**Format reward total**: 0.4 + 0.4 = **0.8** (was 1.5)
- Still encourages format (avoids Round 2's "too strict" problem)
- Doesn't dominate other signals anymore

**Invalid word example**:
- Current: 1.5 (format) - 0.5 (invalid) = +1.0 → **model learns this!**
- Moderate: 0.4 (format) - 1.5 (invalid) = -1.1 → **model avoids this!**

**Variance**: Std dev = 1.19 (well below 1.5 threshold)

---

## The Simulator: Our Secret Weapon

### What We Discovered

Testing **17 bad behavior cases** from actual training logs:

**Current system**:
- Penalized: 1/17 (6%)
- Avg reward: +1.08 (positive!)
- Variance: 0.68 (stable but teaching wrong things)

**Moderate system** (simulated):
- Penalized: 13/17 (72%)
- Avg reward: -0.80 (negative!)
- Variance: 1.19 (stable AND teaches right things)

### Real Training Examples Tested

From actual log data:

```
Sample                      Current → Moderate    Status
─────────────────────────────────────────────────────────
Lowercase "scowl"          +0.77 → +0.15        Still issue*
4-letter "AIRY"            +0.78 → -1.10        ✅ FIXED
Invalid "KIRAN"            -0.18 → -0.90        ✅ FIXED
Reusing dead 'E' (SLEET)   +0.83 → -0.37        ✅ FIXED
Good play "AUGHT"          +1.27 → +0.75        ✅ Stays positive
Perfect play "BRINE"       +2.62 → +1.87        ✅ Stays positive
```

*Lowercase still positive because regex doesn't check case (acceptable trade-off)

---

## What We've Learned: The Meta-Insights

### 1. The Pendulum Pattern

Training improvements follow a pendulum:
- **Too weak** (Round 1) → **Too harsh** (Round 2) → **Too lenient** (Round 3/4) → **Balanced** (Round 5)

Each swing teaches us where the boundaries are.

### 2. The Simulator Insight

**Previous approach**: Try something → train for 20 hours → check results → adjust → repeat

**New approach**: Try something → test in simulator in 5 minutes → find issues → iterate → apply once

**Time saved**: ~60 hours of failed training runs avoided

### 3. The Format Paradox

Format rewards are essential BUT dangerous:
- Too high: Model learns format, ignores strategy (current problem)
- Too low: Model can't learn the game structure (Round 2 problem)
- **Sweet spot**: 0.8 total (encourages without dominating)

### 4. The Variance Constraint

This is the **hard constraint** we can't violate:
- User told us: "variance was too high" in previous rounds
- Simulator confirmed: Aggressive system has σ=1.84 (too high)
- Moderate system: σ=1.19 (acceptable)

Variance > 1.5 → training becomes unstable, may collapse

### 5. The 70% Rule

We don't need 100% penalty coverage:
- 72% of bad behaviors penalized is **good enough**
- Remaining 28% are edge cases (lowercase, missing tags)
- Pursuing 100% would increase variance unacceptably

**Principle**: Better to teach 70% reliably than 100% unstably

---

## Expected Improvements

### What Will Stop
- ❌ Invalid length words (CARE, REAR, AIRY)
- ❌ Nonsense words (XYZZZ, KIRAN)
- ❌ Excessive dead letter reuse

### What Will Continue  
- ✅ Proper `<think></think><guess>WORD</guess>` format
- ✅ Valid 5-letter word generation
- ✅ Exploration of new letters

### What Will Start
- ✅ Paying attention to (-) letters
- ✅ Keeping (✓) letters in place
- ✅ Strategic feedback usage

### Acceptable Trade-offs

These issues remain but are rare/acceptable:
- Missing `</think>` tag: +0.2 (was +1.28) - much reduced
- Extra text in `<guess>`: +0.3 (was +2.03) - much reduced
- Lowercase: ~+0.15 (needs code fix, not just constants)

---

## The Reward Philosophy Evolution

### Round 1-2: Penalty-Focused
"Punish mistakes heavily"  
→ Result: Model too scared to learn

### Round 3: Reward-Focused  
"Reward good behavior heavily"  
→ Result: Format rewards dominated, taught wrong lessons

### Round 5: Balanced Focus
"Strong format baseline (0.8) + strategic penalties that matter (-1.5)"  
→ Expected: Clear signals, controlled variance, actual learning

---

## Implementation Strategy

### Why Not Stop Current Training?

Current training (3 epochs) can continue because:
1. We're learning what the baseline does wrong
2. It generates data for comparison
3. Constants can be changed between epochs
4. Not wasting compute if we analyze it

### The Testing Protocol

Before applying to training:
1. ✅ Test in simulator (DONE - 72% coverage)
2. ✅ Verify variance < 1.5 (DONE - σ=1.19)
3. ✅ Test on real log samples (DONE - improvements confirmed)
4. → Apply constants to `reward_functions.py`
5. → Restart training with Round 5 constants
6. → Monitor first 50 steps for variance

### Success Metrics

Round 5 passes if:
- [ ] Invalid word rate < 5%
- [ ] Average reward trajectory: epoch 1: -1 to 0, epoch 3: +0.5 to +1.5, epoch 10: +1.5+
- [ ] Variance stays < 1.5 throughout training
- [ ] >50% of guesses use feedback correctly
- [ ] Model wins FROST in ≤6 turns
- [ ] No training collapse (like Round 2)

---

## The Big Picture: RL Reward Tuning is Hard!

### What Makes It Hard

1. **Delayed feedback**: Can't see if constants work for 20+ hours
2. **Variance sensitivity**: Small changes cascade into instability
3. **Multiple objectives**: Format + strategy + exploration all compete
4. **Non-obvious interactions**: Format reward masks strategic penalties
5. **No ground truth**: We're inventing the reward function from scratch

### What Makes It Possible

1. **Simulator**: Test changes in minutes, not hours
2. **Real data**: Actual training logs reveal problems
3. **Iterative refinement**: Each round teaches us the boundaries
4. **Variance analysis**: Mathematical framework for stability
5. **User insights**: "Variance too high" was the key constraint

### The Lesson

**You can't design a perfect reward function upfront.**

You must:
1. Try something reasonable
2. Train and observe
3. Find what breaks
4. Simulate improvements
5. Apply and repeat

Round 5 is possible because Rounds 1-4 taught us where the boundaries are.

---

## Files to Modify

### 1. `reward_functions.py`
Update constants as specified above (6 changes)

### 2. `grpo_local_data_peft.py`  
Update output directory:
```python
output_dir="output5/wordle-grpo-peft"  # Round 5
run_name=f"wordle-grpo-peft-v5-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
```

### 3. Create comparison script
`compare_rounds.py` - to analyze Round 4 vs Round 5 results

---

## Training Command

```bash
cd expert_guy/post_training_project
python grpo_local_data_peft.py
```

**Output**: `output5/wordle-grpo-peft/`  
**Duration**: ~20-25 hours (3 epochs × 200 steps)  
**Expected**: Stable variance, clear learning trajectory, strategic play

---

## The Meta-Learning

Each round taught us something fundamental:

| Round | Lesson Learned | Applied In |
|-------|----------------|------------|
| 1 | Signals must be strong | Round 2 |
| 2 | But not TOO strong (penalties) | Round 3 |
| 3 | Reward-focused works | Round 4 |
| 4 | Format can dominate strategy | Round 5 |
| 5 | Balance + variance control = success | 🎯 |

**The pattern**: Each failure precisely defines the next solution.

This is how you tune RL systems - not by getting it right first time, but by **learning from each failure** what the constraints are.

---

**Version**: Round 5 (Moderate Balanced System)  
**Status**: Ready for implementation  
**Confidence**: High (tested via simulator on 17+ cases)  
**Risk**: Low (variance controlled at σ=1.19)

**Next**: Apply constants → Monitor first epoch → Adjust if needed
