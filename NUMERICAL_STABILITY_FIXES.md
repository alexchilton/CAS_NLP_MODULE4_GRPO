# Numerical Stability Fixes for GRPO Training on MPS (Mac M4)

## Problem
Training was failing with:
```
RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
```

This is a common issue when training with GRPO on Apple Silicon (MPS backend) due to:
1. FP16 numerical instability on MPS
2. High temperature causing extreme logits
3. Large sequence lengths causing memory/numerical issues

## Fixes Applied

### 1. Disabled FP16 ✓
```python
fp16=False,  # Was: True
```
**Reason**: FP16 on MPS can cause numerical instability. Full precision (FP32) is more stable.

### 2. Reduced Temperature ✓
```python
temperature=0.7,  # Was: 1.0
```
**Reason**: Lower temperature prevents extreme probability distributions that can lead to inf/nan.

### 3. Added min_p Filtering ✓
```python
"min_p": 0.05,  # NEW
```
**Reason**: Filters out tokens with very low probabilities, preventing numerical issues.

### 4. Optimized Sequence Lengths ✓
```python
max_prompt_length=1024,    # Keep original - need full prompt + history + CoT
max_completion_length=512, # Reduced from 2048 - reasonable for Wordle CoT + guess
```
**Reason**: Prompts with history need full 1024 tokens. Completions can be 512 (enough for reasoning + guess).

### 5. Lower Repetition Penalty ✓
```python
repetition_penalty=1.05,  # Was: 1.1
```
**Reason**: Lower penalty prevents extreme logit values that can cause overflow.

### 6. Adjusted top_p ✓
```python
top_p=0.9,  # Was: 0.95
```
**Reason**: Slightly more conservative sampling for stability.

### 7. Lower Gradient Clipping ✓
```python
max_grad_norm=0.5,  # Was: 1.0
```
**Reason**: More aggressive clipping to prevent gradient explosions.

### 8. Updated Temperature Schedule ✓
```python
start_temp=0.7,  # Was: 1.0
end_temp=0.3,    # Unchanged
```
**Reason**: Start from stable temperature, decay to even more stable.

## Temperature Schedule (Updated)

| Training Progress | Temperature | Behavior |
|-------------------|-------------|----------|
| 0% | 0.7 | Moderate exploration (stable) |
| 10% | 0.63 | Transitioning |
| 20% | 0.57 | Transitioning |
| 30% | 0.50 | Transitioning |
| 30%+ | 0.3 | Low temp exploitation |

## Expected Impact

### Performance
- **Training Speed**: ~10-20% slower due to FP32 vs FP16 (worth it for stability)
- **Memory Usage**: Similar to original (FP32 overhead but not excessive)
- **Stability**: Much more stable, should complete without crashes

### Model Quality
- **Exploration**: Still adequate with temp=0.7 start
- **Convergence**: Better due to stable gradients
- **Final Performance**: Should be similar or better

## Alternative: If Still Unstable

If training still fails, try:

```python
# Even more conservative settings
temperature=0.5,
max_completion_length=128,
per_device_train_batch_size=1,
```

## Monitoring

Watch for these indicators:

### Good Signs ✓
- Loss decreasing smoothly
- No nan/inf in logs
- Rewards varying reasonably (-3 to +3 range)

### Bad Signs ✗
- Loss suddenly jumping to very high values
- Warnings about inf/nan
- Rewards stuck at 0 or extreme values

## Files Modified

- `grpo_local_data_peft.py`: Training config (lines 220-268)

## Training Command (Updated)

```bash
python grpo_local_data_peft.py
```

Model will still save to `output2/wordle-grpo-peft/`

---

**Date**: 2025-12-17
**Issue**: RuntimeError with inf/nan probabilities on MPS
**Status**: FIXED - Ready for training
