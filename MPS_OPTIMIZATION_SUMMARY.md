# MPS/Metal Training Optimization Summary

**Date**: 2025-12-18
**Problem**: 226s/iteration on 3B + LoRA (way too slow)
**Target**: <40s/iteration (practical for 2000 steps)

---

## The Problem

At 226s/iteration:
- 120 steps → ~7 hours
- 2000 steps → ~5 days ❌ **IMPRACTICAL**

**Root cause**: Metal unified memory swapping to disk under memory pressure

---

## Optimization Strategy Applied

### ✅ Changes Made to `grpo_local_data_peft.py`

| Parameter | Before | After | Impact |
|-----------|--------|-------|--------|
| **max_completion_length** | 512 | **128** | 🔥 **4x memory reduction** |
| **gradient_checkpointing** | False | **True** | 🔥 **~30% less memory** |
| **per_device_train_batch_size** | 2 | **1** | 💧 Less memory pressure |
| **gradient_accumulation_steps** | 1 | **2** | ⚖️ Maintains effective batch=2 |
| **save_steps** | 20 | **50** | 💨 Less I/O overhead |
| **save_total_limit** | 5 | **3** | 💾 Less disk usage |
| **low_cpu_mem_usage** | - | **True** | 🧠 Lower peak load |
| **MPS cache clearing** | - | **Added** | 🗑️ Pre-emptive cleanup |

### 🎯 Expected Results

**Conservative estimate**: 226s → **60-80s/iteration**
**Optimistic estimate**: 226s → **35-50s/iteration**

At 60s/iteration:
- 120 steps → 2 hours ✅
- 2000 steps → 33 hours ✅ **PRACTICAL**

---

## Why These Changes Work

### 1. **max_completion_length: 512 → 128** (BIGGEST WIN)

**Problem**: Wordle reasoning + guess needs ~80 tokens max, not 512!

**Impact**:
- Generation is O(n²) in sequence length
- 512 → 128 = 4x fewer tokens = ~16x less compute
- Memory: 4x reduction in KV cache size

**Real training examples** (from your logs):
```
✅ Good: "<think>FROST: F(-), R(✓), O(~), S(-), T(-)...</think><guess>BRINE</guess>" = ~75 tokens
❌ Before: Allocated 512 tokens (437 wasted!)
```

### 2. **gradient_checkpointing: True** (CRITICAL FOR MPS)

**What it does**: Recomputes activations during backward pass instead of storing them

**Trade-off**:
- Memory: 30-50% reduction ✅
- Compute: ~20% slower per step
- **Net**: Prevents swapping = 3-5x faster overall ✅

**Why essential on Metal**: Unified memory means swapping kills performance

### 3. **Batch size: 2 → 1 (with grad_accum: 1 → 2)**

**Problem**: MPS kernel launch overhead is high, large batches amplify it

**Solution**:
- Smaller batches = less peak memory
- Gradient accumulation maintains effective batch size
- Same learning dynamics, less memory pressure

### 4. **MPS-specific optimizations**

```python
# Added to model loading
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# Added to model loading
low_cpu_mem_usage=True
```

**Impact**: Prevents allocation fragmentation, reduces peak memory

---

## What the OTHER Advice Got Wrong

### ❌ **8-bit quantization with BitsAndBytes**

**Advice**: "Use `load_in_8bit=True`"
**Reality**: BitsAndBytes requires CUDA, doesn't work on MPS
**Status**: Skip entirely

### ❌ **`device_map="mps"`**

**Advice**: "Set `device_map='mps'`"
**Reality**: PyTorch doesn't support this, use `.to("mps")` or `device_map="auto"`
**Status**: We use `device_map="auto"` (correct)

### ❌ **`torch.mps.set_per_process_memory_fraction()`**

**Advice**: "Allocate 80% to MPS"
**Reality**: This API doesn't exist for MPS (CUDA only)
**Status**: Skip

### ⚠️ **`sudo purge`**

**Advice**: "Clear macOS file cache"
**Reality**: Might help once, but doesn't address root cause
**Status**: Can try but focus on above changes first

---

## What We KEPT from the Advice

### ✅ **Reduce rollout length**

Already doing this:
- `max_completion_length=128` (was the key insight!)
- `num_generations=2` (already at minimum)

### ✅ **Micro-batching strategy**

Applied:
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=2`

---

## Testing Protocol

### Before Training
```bash
# Clear MPS cache (one-time)
python -c "import torch; torch.mps.empty_cache() if torch.backends.mps.is_available() else None"

# Optional: Clear macOS cache
sudo purge
```

### Start Training
```bash
cd expert_guy/post_training_project
python grpo_local_data_peft.py
```

### Monitor Performance

**First 10 iterations**: Check s/it in logs
- Target: <80s/iteration
- Ideal: <50s/iteration

**Memory monitoring**:
```bash
# In another terminal
watch -n 5 'ps aux | grep python | grep -v grep'
```

Look for:
- Resident set size (RSS) should be stable
- No excessive swapping (monitor Activity Monitor)

### Success Metrics

- [ ] Iteration time < 80s (conservative target)
- [ ] Iteration time < 50s (optimistic target)
- [ ] 2000 steps completes in < 48 hours
- [ ] No memory warnings in logs
- [ ] Training doesn't crash

---

## Rollback Plan

If iteration time is still >100s after 20 steps:

### Extreme option 1: Reduce max_new_tokens further
```python
max_completion_length=96,  # From 128
```

### Extreme option 2: Single rollout for first 50 steps
```python
num_generations=2,  # Can't go lower (GRPO minimum)
```

### Extreme option 3: Skip info_gain for first 200 steps
Modify `wordle_reward_func` to conditionally disable `guess_value()` (adds second forward pass)

---

## The Math: Why 4x Matters

**Memory for attention** = O(batch × seq_len × hidden_dim)

Before:
- batch=2, seq=512, hidden=2560 (Qwen-3B)
- Attention memory: 2 × 512 × 2560 = 2.6M floats = 10.4 MB per layer
- 36 layers = 374 MB just for attention activations

After:
- batch=1, seq=128, hidden=2560
- Attention memory: 1 × 128 × 2560 = 327k floats = 1.3 MB per layer
- 36 layers = 47 MB for attention activations

**Reduction**: 374 MB → 47 MB = **8x less memory**

With gradient checkpointing + smaller batches + shorter sequences:
**Total memory reduction**: ~5-6x

**Effect**: Eliminates swapping → 3-5x faster iteration

---

## Expected Timeline

**Old (226s/it)**:
- 2 epochs × ~120 steps = 240 steps
- 240 × 226s = 15 hours per run

**New (60s/it estimate)**:
- 2 epochs × ~120 steps = 240 steps
- 240 × 60s = 4 hours per run ✅

**New (40s/it optimistic)**:
- 2 epochs × ~120 steps = 240 steps
- 240 × 40s = 2.6 hours per run 🎉

---

## Key Insights

### Why the Original Advice Was Half-Right

✅ **Correct diagnosis**: Memory pressure causing swaps
✅ **Correct insight**: Reduce max_new_tokens
❌ **Wrong solutions**: APIs that don't exist on MPS
❌ **Wrong priority**: Quantization before sequence length

### The Real Optimization Order

1. **Sequence length** (4x impact) ← WE DID THIS
2. **Gradient checkpointing** (2x impact) ← WE DID THIS
3. **Batch size** (1.5x impact) ← WE DID THIS
4. **MPS cache management** (1.2x impact) ← WE DID THIS
5. ~~Quantization (can't do on MPS anyway)~~

---

## Files Modified

1. **grpo_local_data_peft.py**
   - Lines 127-148: Added MPS cache clearing
   - Lines 224-244: Optimized training config
   - Lines 248-250: Updated output directories

---

## Next Steps

1. **Run training** with optimized config
2. **Monitor first 20 iterations** for speed
3. **Compare to baseline**: Should see 3-4x speedup
4. **If still slow**: Consider extreme options above
5. **Once stable**: Focus on Round 5 reward improvements

---

**Version**: MPS Optimization v1
**Status**: Ready to test
**Expected improvement**: 226s → 40-80s per iteration
**Confidence**: High (based on standard MPS optimization patterns)

**Key takeaway**: The 512 token limit was the silent killer. Wordle doesn't need 512 tokens!
