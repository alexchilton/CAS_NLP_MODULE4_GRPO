# Final Training Configuration - 70 Epochs for 2100 Steps

**Date**: 2025-12-18
**Critical Discovery**: Dataset only has 60 training samples!
**Solution**: Run 70 epochs to reach 2100 steps minimum

---

## The Step Count Problem

### Initial Misunderstanding
```
❌ Thought: "RL needs 5-10 epochs"
❌ Reality: RL needs 1000-2000+ STEPS (not epochs!)
```

### Actual Dataset Size
```
Total valid samples: 75
Train samples: 60
Val samples: 15

Steps per epoch: 60 samples ÷ 2 batch size = 30 steps
```

### Why 2000+ Steps Required (from expert advice)

> "120 GRPO steps is almost certainly too small for the model to get past the 'imitate the template' stage"

**Learning phases**:
- **< 500 steps**: Only format converges (template imitation)
- **1000-2000 steps**: Qualitative jump - starts using feedback strategically
- **5000-10000 steps**: Super-human play (optimal entropy, 3-4 guess solves)

### Calculation
```
Minimum target: 2000 steps
Steps per epoch: 30
Required epochs: 2000 ÷ 30 = 66.7 epochs

Selected: 70 epochs = 2100 steps ✅
```

---

## Final Configuration

```python
num_train_epochs=70                    # 70 × 30 = 2100 steps
per_device_train_batch_size=2          # Must be divisible by num_generations
per_device_eval_batch_size=2           # Must match train batch
gradient_accumulation_steps=1          # Can't reduce batch (GRPO constraint)
num_generations=2                      # Minimum for GRPO

max_completion_length=128              # 🔥 4x memory reduction (was 512)
gradient_checkpointing=True            # 🔥 30% memory reduction (was False)

logging_steps=10                       # Every 10 steps
eval_steps=100                         # Every 100 steps (21 evals total)
save_steps=100                         # Every 100 steps (must match eval_steps)
save_total_limit=10                    # Keep last 1000 steps of checkpoints
```

---

## Optimization Applied

| Parameter | Before | After | Impact |
|-----------|--------|-------|--------|
| **max_completion_length** | 512 | **128** | 🔥 4x memory |
| **gradient_checkpointing** | False | **True** | 🔥 30% memory |
| **num_train_epochs** | 2 | **70** | 🎯 Reaches 2100 steps |
| **logging_steps** | 1 | **10** | Less I/O overhead |
| **eval_steps** | 20 | **100** | Appropriate for 2100 steps |
| **save_steps** | 20 | **100** | Less disk I/O |

**Note**: Wanted `batch_size=1` for more memory savings, but GRPO requires batch divisible by `num_generations=2`.

---

## Expected Training Timeline

### With Optimizations (targeting 40-80s/iteration)

**Total steps**: 2100
**Checkpoints**: Every 100 steps (21 checkpoints)
**Evaluations**: Every 100 steps (21 evals)

| Scenario | Time per Step | Total Duration |
|----------|---------------|----------------|
| **Optimistic** | 40s | 23.3 hours |
| **Conservative** | 60s | 35 hours |
| **Worst case** | 80s | 46.7 hours |

**Target**: Complete in 24-36 hours

---

## Learning Milestones to Watch For

### Steps 0-500: Format Learning
**Metrics**:
- Valid format rate: 0% → 95%
- Average reward: -0.5 → +0.3
- Invalid word rate: 40% → 10%

**Checkpoint**: Step 500
**Expected**: `running_mean(total_reward) ≈ +0.3`

### Steps 500-1000: Strategy Emergence 🎯 CRITICAL
**Metrics**:
- Feedback usage: 10% → 40%
- Dead letter reuse: 60% → 30%
- Average reward: +0.3 → +1.0

**Checkpoint**: Step 1000
**Expected**: `running_mean(feedback_reward) > 0.0` (first time!)

### Steps 1000-1500: Strategic Play
**Metrics**:
- Feedback usage: 40% → 60%
- Average reward: +1.0 → +1.5
- Win rate: 30% → 60%

**Checkpoint**: Step 1500
**Expected**: `running_mean(total_reward) > 1.5`

### Steps 1500-2100: Refinement
**Metrics**:
- Win rate: 60% → 70%+
- Average reward: +1.5 → +2.0
- Consistent strategic play

**Checkpoint**: Step 2100 (final)
**Expected**: Solves FROST in ≤6 turns

---

## Success Criteria (from expert advice)

**Training passes if BOTH conditions hold**:
```python
running_mean(total_reward) > 1.5
AND
running_mean(feedback_reward) > 0.0
```

**What this means**:
- Model respects the ✓/×/~ feedback mask
- Actually plays Wordle strategically
- Not just "pretty-printing" format

**Earliest useful checkpoint**: When both conditions first met (likely around step 1000-1500)

---

## Why We Couldn't Reduce Batch Size Further

### GRPO Constraint
```python
# GRPO validation requires:
(per_device_batch_size * num_devices) % num_generations == 0

# Our setup:
batch_size=2, num_devices=1, num_generations=2
2 * 1 % 2 = 0 ✅

# Tried:
batch_size=1, num_devices=1, num_generations=2
1 * 1 % 2 = 1 ❌ "must be divisible"
```

**Can't change**:
- `num_generations=2` is minimum for GRPO (needs comparisons)
- `num_devices=1` (single GPU/MPS)

**Must keep**:
- `batch_size=2`

**Still got huge wins from**:
- `max_completion_length`: 512 → 128 (4x memory)
- `gradient_checkpointing`: True (30% memory)

---

## Memory Breakdown

### Before Optimizations
```
Batch: 2 samples
Sequence: 512 tokens
Activations: ~2GB per forward pass
Swapping to disk: YES → 226s/iteration
```

### After Optimizations
```
Batch: 2 samples (unchanged, GRPO constraint)
Sequence: 128 tokens (4x smaller!)
Activations: ~500MB per forward pass
Gradient checkpointing: Recompute instead of store
Swapping: Minimal/None → 40-80s/iteration
```

**Expected speedup**: 3-5x (226s → 40-80s)

---

## Monitoring Plan

### First 50 Steps (critical validation)
**Watch for**:
- Iteration speed < 80s? ✅ Optimizations working
- No memory errors? ✅ Stable
- Rewards not collapsing? ✅ Training viable

**If any fail**: Stop and debug immediately

### Step 500 (format checkpoint)
**Check**:
- Valid format rate > 90%?
- Rewards positive on average?
- No training collapse?

### Step 1000 (strategy checkpoint) 🎯
**Check**:
- `feedback_reward > 0.0`? ← **KEY MILESTONE**
- Total reward > +1.0?
- Model using feedback mask?

**If not met**: Training may need more steps or reward tuning

### Step 1500 (performance checkpoint)
**Check**:
- `total_reward > 1.5`? ← **SUCCESS CRITERIA**
- Win rate > 50%?
- Consistent strategic play?

### Step 2100 (final)
**Evaluate**:
- Test on FROST puzzle
- Check win rate on held-out set
- Verify strategic reasoning

---

## Expected Outcome

### If Successful
**Checkpoint**: ~step 1000-1500
**Behavior**:
- ✅ Proper format always
- ✅ Respects feedback (✓/×/~)
- ✅ Strategic letter exploration
- ✅ Wins Wordle consistently in ≤6 turns

### If Needs More
**Checkpoint**: Step 2100 still improving
**Action**: Extend to 3000-5000 steps
**Time**: Add another 20-40 hours

### If Plateaus
**Symptom**: Flat reward curve for 500+ steps
**Likely cause**: Reward function needs tuning
**Action**: Stop, analyze, adjust reward constants

---

## Files Modified

✅ `grpo_local_data_peft.py`:
- Line 226: `num_train_epochs=70` (to reach 2100 steps)
- Line 227-229: `batch_size=2, grad_accum=1` (GRPO constraint)
- Line 242: `max_completion_length=128` (4x reduction)
- Line 244: `gradient_checkpointing=True` (enabled)
- Line 232-237: Adjusted logging/eval/save for long run

---

## Start Training

```bash
cd /Users/alexchilton/Downloads/Current_Learning/uni/BERN/module4_transformer_grpo/expert_guy/post_training_project
python grpo_local_data_peft.py
```

**Expected duration**: 24-36 hours
**Output**: `output5/wordle-grpo-optimized/`
**Checkpoints**: 21 checkpoints (every 100 steps)
**Best model**: Auto-saved based on eval_loss

---

## Key Insights

1. **Steps matter, not epochs** - With small datasets, need many epochs
2. **2000+ steps minimum** - Below this, only format converges
3. **GRPO constraints** - Batch size must be divisible by num_generations
4. **Sequence length is key** - 512→128 gives 4x memory savings
5. **Be patient** - Real learning happens at steps 1000-2000

---

**Status**: ✅ Ready to train
**Total steps**: 2100 (exceeds 2000 minimum)
**Optimizations**: max_completion_length + gradient_checkpointing
**Expected**: 3-5x speedup, strategic Wordle play by step 1500
