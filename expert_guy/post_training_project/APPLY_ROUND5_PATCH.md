# How to Apply Round 5 Moderate System Patch

**Created**: 2025-12-18  
**Status**: Ready to apply when current training completes  
**File**: `reward_functions_round5.py` (new file, won't interfere with running training)

---

## What's Been Prepared

✅ **reward_functions_round5.py** - Complete new reward functions with Round 5 constants  
✅ **ROUND5_IMPROVEMENTS.md** - Full documentation of changes and rationale  
✅ **ROUND5_FAILURE_PREDICTIONS.md** - Expected failure modes and fixes  
✅ **Verification script** - Confirmed simulator values match documentation

---

## When to Apply

**Option 1: After current training completes** (Recommended)
- Let current Round 4 training finish (still running)
- Compare Round 4 final results to baseline
- Then apply Round 5 for fresh start

**Option 2: Stop current and apply now**
- If current training shows no improvement
- If you want to test Round 5 sooner
- Saves compute if Round 4 is clearly broken

---

## How to Apply (Step by Step)

### 1. Backup Current File
```bash
cd expert_guy/post_training_project
cp reward_functions.py reward_functions_round4_backup.py
```

### 2. Apply the Patch
```bash
# Replace current with Round 5 version
cp reward_functions_round5.py reward_functions.py
```

### 3. Verify Changes Applied
```bash
python3 << 'PYEOF'
from reward_functions import (
    VALID_FORMAT_REWARD,
    VALID_WORD_BONUS,
    INVALID_LENGTH_PENALTY,
    INVALID_WORD_PENALTY,
    DEAD_LETTER_PENALTY,
    MISSING_GOOD_LETTER_PENALTY
)

expected = {
    "VALID_FORMAT_REWARD": 0.4,
    "VALID_WORD_BONUS": 0.4,
    "INVALID_LENGTH_PENALTY": -1.5,
    "INVALID_WORD_PENALTY": -1.5,
    "DEAD_LETTER_PENALTY": -0.7,
    "MISSING_GOOD_LETTER_PENALTY": -0.6,
}

actual = {
    "VALID_FORMAT_REWARD": VALID_FORMAT_REWARD,
    "VALID_WORD_BONUS": VALID_WORD_BONUS,
    "INVALID_LENGTH_PENALTY": INVALID_LENGTH_PENALTY,
    "INVALID_WORD_PENALTY": INVALID_WORD_PENALTY,
    "DEAD_LETTER_PENALTY": DEAD_LETTER_PENALTY,
    "MISSING_GOOD_LETTER_PENALTY": MISSING_GOOD_LETTER_PENALTY,
}

print("Verifying Round 5 constants:")
all_match = True
for key in expected:
    if abs(expected[key] - actual[key]) < 0.01:
        print(f"  ✅ {key}: {actual[key]}")
    else:
        print(f"  ❌ {key}: Expected {expected[key]}, got {actual[key]}")
        all_match = False

if all_match:
    print("\n✅ All Round 5 constants applied correctly!")
else:
    print("\n❌ Some constants don't match - check the patch")
PYEOF
```

### 4. Update Training Script (Optional)
If you want to track this as Round 5:

```bash
# Edit grpo_local_data_peft.py
# Change line ~222:
output_dir="output5/wordle-grpo-peft"  # was output4
# Change line ~247:
run_name=f"wordle-grpo-peft-v5-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
```

### 5. Start Round 5 Training
```bash
cd expert_guy/post_training_project
python grpo_local_data_peft.py
```

---

## What Changed (Summary)

| Constant | Round 4 | Round 5 | Change |
|----------|---------|---------|--------|
| VALID_FORMAT_REWARD | 1.0 | 0.4 | -60% 📉 |
| PARTIAL_FORMAT_REWARD | 0.3 | 0.2 | -33% 📉 |
| VALID_WORD_BONUS | 0.5 | 0.4 | -20% 📉 |
| INVALID_LENGTH_PENALTY | -0.3 | -1.5 | -400% 📉 |
| INVALID_WORD_PENALTY | -0.5 | -1.5 | -200% 📉 |
| DEAD_LETTER_PENALTY | -0.4 | -0.7 | -75% 📉 |
| MISSING_GOOD_LETTER_PENALTY | -0.3 | -0.6 | -100% 📉 |

**Impact**:
- Format total: 1.5 → 0.8 (won't dominate anymore)
- Invalid word: +0.72 → -1.1 (now properly penalized)
- Dead letter reuse: +0.38 → -0.4 (now properly penalized)
- Variance: σ = 1.19 (controlled, below 1.5 threshold)

---

## Monitoring Checklist (First 24 Hours)

### First 10 Steps (Critical)
- [ ] Check if `<guess>` tags appear
- [ ] Verify variance < 2.0
- [ ] Look for any positive rewards
- [ ] No immediate training collapse

**If problems**:
- See ROUND5_FAILURE_PREDICTIONS.md for diagnosis

### First 50 Steps
- [ ] Reward trend upward (even slightly)
- [ ] Invalid word rate decreasing
- [ ] No "lowercase loop" emerging
- [ ] Variance staying < 1.5

### First Epoch
- [ ] Compare to Round 4 baseline
- [ ] Check reward distribution
- [ ] Sample outputs manually
- [ ] Decide: continue or adjust

---

## Rollback Plan (If Needed)

If Round 5 fails catastrophically:

```bash
# Restore Round 4
cp reward_functions_round4_backup.py reward_functions.py

# Or try Round 5.1 (pre-designed fixes in ROUND5_FAILURE_PREDICTIONS.md)
```

---

## Success Indicators

Round 5 is working if you see:

✅ **First 10 steps**: Format rewards 0.4-0.8, some strategic penalties  
✅ **First 50 steps**: Mean reward trending from -1 toward 0  
✅ **First epoch**: Mean reward > -0.5, variance < 1.5  
✅ **By epoch 3**: Mean reward > 0, invalid rate < 10%

---

## Files Summary

**New files created (safe, won't interfere with running training)**:
- `reward_functions_round5.py` - Round 5 patch
- `ROUND5_IMPROVEMENTS.md` - Full documentation
- `ROUND5_FAILURE_PREDICTIONS.md` - Expected issues
- `APPLY_ROUND5_PATCH.md` - This file
- `FINAL_RECOMMENDATION.md` - Moderate system details
- `REWARD_SYSTEM_SUMMARY.md` - Problem analysis
- Various test scripts in current directory

**Files to modify when applying**:
- `reward_functions.py` (replace with Round 5 version)
- `grpo_local_data_peft.py` (optional: update output_dir to output5)

**Files to backup**:
- `reward_functions.py` → `reward_functions_round4_backup.py`

---

## Quick Apply Command (When Ready)

```bash
cd expert_guy/post_training_project

# Backup current
cp reward_functions.py reward_functions_round4_backup.py

# Apply patch
cp reward_functions_round5.py reward_functions.py

# Verify
python -c "from reward_functions import VALID_FORMAT_REWARD; print('✅ Round 5 applied!' if VALID_FORMAT_REWARD == 0.4 else '❌ Not applied')"

# Start training
python grpo_local_data_peft.py
```

---

**Ready to go when you are!** 🚀

Let the current training finish, then apply Round 5 tomorrow and see where it breaks! 🔬
