# FINAL REWARD TUNING RECOMMENDATION

## Context
- **Previous version:** Too strict formatting → couldn't learn game
- **Current version:** Too lenient → learns format but ignores strategy
- **Constraint:** Keep variance low (std < 1.5) for stable gradients

## The Three Systems Compared

| Metric | Current | Aggressive | **Moderate** | Target |
|--------|---------|------------|--------------|--------|
| Std Dev | 0.68 ✅ | 1.84 ❌ | **1.19 ✅** | < 1.5 |
| Separation | 1.12 ❌ | 3.38 ✅ | **2.53 ✅** | > 2.0 |
| Penalties | 0% ❌ | 61% ⚠️ | **72% ✅** | > 70% |

## Recommended Changes (Moderate System)

### Constants to Update

```python
# reward_functions.py

# FORMAT REWARDS (slight reduction from current)
VALID_FORMAT_REWARD = 0.4          # was 1.0 → reduce 60%
PARTIAL_FORMAT_REWARD = 0.2        # was 0.3 → reduce 33%
VALID_WORD_BONUS = 0.4             # was 0.5 → reduce 20%

# PENALTIES (3-5x increase)
INVALID_LENGTH_PENALTY = -1.5      # was -0.3 → 5x stronger
INVALID_WORD_PENALTY = -1.5        # was -0.5 → 3x stronger
DEAD_LETTER_PENALTY = -0.7         # was -0.4 → 1.75x stronger
MISSING_GOOD_LETTER_PENALTY = -0.6 # was -0.3 → 2x stronger

# Keep others the same
CORRECT_POSITION_REWARD = 0.4
NEW_POSITION_REWARD = 0.3
REPEATED_WRONG_POSITION = -0.2
EXPLORATION_BONUS = 0.05
```

## Why This Works

### 1. Controlled Variance (std = 1.19)
- Reward range: -2.0 to +2.5 (~4.5 point spread)
- Gradients will be stable
- Model can learn consistently

### 2. Clear Signal Separation (+2.53 difference)
- Good play: avg +1.73
- Bad play: avg -0.80
- Model clearly knows what to reinforce vs avoid

### 3. Format Still Encouraged
- Valid formatted guess: +0.8 (0.4 + 0.4)
- Not too high to dominate (was +1.5)
- Not too low to discourage (previous issue)

### 4. Strategic Penalties Clear
- Invalid word: -1.5 (was +0.72) → 2.22 point swing
- Reusing dead letters: -1.5 (was +0.38) → 1.88 point swing
- Ignoring feedback: -0.5 (was +1.61) → 2.11 point swing

## Expected Behavior Changes

### What model will STOP doing:
- ❌ Invalid length words (CARE, REAR)
- ❌ Nonsense words (XYZZZ)
- ❌ Reusing dead letters excessively

### What model will CONTINUE doing:
- ✅ Using proper <think></think><guess></guess> format
- ✅ Generating valid 5-letter words
- ✅ Exploring new letters

### What model will START doing:
- ✅ Paying attention to feedback
- ✅ Using (-) letters in new positions
- ✅ Keeping (✓) letters in place

## Remaining Issues (Acceptable Trade-offs)

These behaviors still get slight positive rewards but are rare:
- Missing </think> tag: +0.2 (was +1.28)
- Extra text in guess: +0.3 (was +2.03)

These are acceptable because:
1. Format is still encouraged overall
2. Variance stays controlled
3. Strategic play is now properly rewarded

## Implementation

### Quick Fix (5 minutes):
Update the 6 constants above in `reward_functions.py`

### Testing:
```bash
python test_reward_simulator.py
# Verify:
# - Invalid words get < -1.0
# - Good play gets > +1.0
# - Std dev < 1.5
```

### Re-run Training:
Current training can continue to finish epoch, then restart with new rewards.

## Success Metrics

After retraining with moderate rewards:
- [ ] <5% invalid length/word guesses
- [ ] >50% of guesses use feedback correctly  
- [ ] Model win rate improves
- [ ] Training loss stable (no spikes from variance)

---

**BOTTOM LINE:** Moderate system gives 72% penalty coverage with controlled variance.  
This is the sweet spot between "too strict" (previous) and "too lenient" (current).
