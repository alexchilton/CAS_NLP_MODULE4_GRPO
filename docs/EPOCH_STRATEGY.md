# GRPO Epoch Strategy for Wordle Training

**Date**: 2025-12-18
**Decision**: Run 7 epochs (not 2)
**Rationale**: RL requires more epochs than SFT for strategy emergence

---

## Why More Epochs for RL?

### SFT vs GRPO Learning Dynamics

| Training Type | Epochs | Learning Pattern |
|---------------|--------|------------------|
| **SFT** | 1-3 | Imitation learning from fixed examples |
| **GRPO** | 5-15 | Policy exploration + strategy discovery |

### The RL Learning Curve

```
Epoch 1-2:  📝 Format learning, basic patterns
            - Model learns <think></think><guess>WORD</guess> structure
            - Discovers valid 5-letter word constraint
            - Reward: -0.5 to +0.2

Epoch 3-4:  🧠 Strategy emergence  ← CRITICAL PHASE
            - Starts using feedback (✓, ~, -)
            - Avoids dead letters
            - Explores letter combinations
            - Reward: +0.2 to +1.0

Epoch 5-7:  ⚡ Strategic refinement
            - Optimal letter positioning
            - Information-gain maximization
            - Consistent winning strategies
            - Reward: +1.0 to +2.0

Epoch 8-10: 🎯 Mastery (if needed)
            - Fine-tuning edge cases
            - Minimal improvement
            - Risk of overfitting

Epoch 10+:  ⚠️ Diminishing returns
            - Potential dataset memorization
            - No significant improvement
```

---

## Your Previous Plan (from round5_improvements.md)

Line 260 stated success metrics:
> "Average reward trajectory: **epoch 1: -1 to 0**, **epoch 3: +0.5 to +1.5**, **epoch 10: +1.5+**"

**You were already planning for 10 epochs!**

But your config said:
```python
num_train_epochs=2  # ❌ Stopping before strategy emerges!
```

---

## Time Cost Analysis

### With Optimizations (60s/iteration estimate)

**Dataset size**: ~200 steps per epoch (estimate)

| Epochs | Total Steps | Time @ 60s/it | Time @ 40s/it |
|--------|-------------|---------------|---------------|
| 2 | 400 | 6.7 hours | 4.4 hours |
| 5 | 1000 | 16.7 hours | 11.1 hours |
| **7** | **1400** | **23.3 hours** | **15.6 hours** |
| 10 | 2000 | 33.3 hours | 22.2 hours |

**7 epochs = ~20 hours** (overnight + day run)

### Why 7 is the Sweet Spot

✅ **Sufficient**: Covers format → strategy → refinement phases
✅ **Practical**: ~24 hours fits weekend/overnight schedule
✅ **Safe**: Not so long that overfitting is likely
✅ **Aligned**: Matches your original Round 5 plan trajectory
❌ **Not 10+**: Diminishing returns, risk of memorization

---

## Expected Learning Milestones

### Epoch 1-2: Basics
**Metrics**:
- Valid format rate: 70% → 95%
- Average reward: -0.5 → +0.3
- Invalid word rate: 40% → 10%

**Behaviors**:
- ✅ Generates `<think>...</think><guess>WORD</guess>`
- ✅ Mostly 5-letter words
- ❌ Ignores feedback
- ❌ Random guessing

### Epoch 3-4: Strategy Emergence ⭐ KEY PHASE
**Metrics**:
- Feedback usage: 10% → 50%
- Average reward: +0.3 → +1.2
- Dead letter reuse: 60% → 20%

**Behaviors**:
- ✅ Starts using (✓) correct letters
- ✅ Avoids (-) dead letters
- ✅ Repositions (~) wrong-position letters
- ⚠️ Still suboptimal exploration

### Epoch 5-7: Refinement
**Metrics**:
- Win rate (on test set): 40% → 70%
- Average reward: +1.2 → +1.8
- Information gain per guess: Low → Medium

**Behaviors**:
- ✅ Strategic first guess (vowel-rich)
- ✅ Consistent feedback application
- ✅ Efficient letter elimination
- ✅ Wins FROST-level puzzles in ≤6 turns

### Epoch 8+ (Optional Extension)
Only continue if:
- Reward still climbing at epoch 7
- Win rate < 70%
- Strategy not yet consistent

---

## Decision: 7 Epochs

### Configuration Update

```python
num_train_epochs=7  # ✅ CHANGED from 2
```

### Rationale

1. **Minimum viable**: 5 epochs needed for strategy
2. **Buffer**: +2 epochs for refinement
3. **Time feasible**: ~24 hours (practical for Mac)
4. **Aligned with plan**: Matches Round 5 expectations
5. **Not excessive**: 10+ risks overfitting

### Monitoring Plan

**After epoch 2** (checkpoint):
- ✅ Valid format rate > 90%?
- ✅ Rewards trending positive?
- ✅ Training stable (no collapse)?

**After epoch 4** (strategy check):
- ✅ Feedback usage > 40%?
- ✅ Average reward > +0.8?
- ✅ Dead letter reuse < 30%?

**After epoch 7** (final evaluation):
- ✅ Win rate on test set > 60%?
- ✅ Average reward > +1.5?
- ✅ FROST puzzle solved in ≤6 turns?

---

## Comparison: 2 vs 7 Epochs

| Aspect | 2 Epochs | 7 Epochs |
|--------|----------|----------|
| **Format learning** | ✅ Complete | ✅ Complete |
| **Strategy emergence** | ❌ **Incomplete** | ✅ **Complete** |
| **Feedback usage** | ❌ ~20% | ✅ ~60% |
| **Win rate** | ❌ ~30% | ✅ ~70% |
| **Time cost** | 6 hours | 20 hours |
| **Outcome** | Partial success | Full success |

**Verdict**: 2 epochs is **insufficient** for GRPO strategy development

---

## Risk Mitigation

### Risk: Training Collapse
**Probability**: Low (Round 4 ran successfully)
**Mitigation**: Checkpointing every 50 steps
**Recovery**: Resume from last checkpoint

### Risk: Overfitting
**Probability**: Low at 7 epochs
**Detection**: Val loss diverges from train loss
**Mitigation**: Stop early if detected

### Risk: Time Overrun
**Probability**: Medium (if optimization fails)
**Mitigation**: First 20 iterations show speed
**Recovery**: Reduce to 5 epochs if >80s/it

### Risk: Plateau
**Probability**: Medium (reward stops improving)
**Detection**: Flat reward curve for 2+ epochs
**Action**: Stop early, don't waste compute

---

## Alternative Strategies (if needed)

### If Time is Critical
**Option A**: Run 5 epochs minimum
- Still covers format → strategy → basic refinement
- ~14 hours instead of 24 hours
- Acceptable compromise

### If Results Plateau Early
**Option B**: Early stopping
- Monitor reward curve
- If flat for 50 steps → stop
- Save compute for next experiment

### If Training Explodes
**Option C**: Immediate intervention
- If loss spikes or collapse
- Stop, analyze logs
- Adjust reward constants before continuing

---

## Updated Training Timeline

### With 7 Epochs @ 60s/iteration

**Start**: Evening (e.g., 8 PM)
**Epoch 1-2**: 8 PM → 2 AM (6 hours)
**Epoch 3-4**: 2 AM → 8 AM (6 hours)
**Epoch 5-6**: 8 AM → 2 PM (6 hours)
**Epoch 7**: 2 PM → 5 PM (3 hours)
**Complete**: Next day ~5 PM (21 hours total)

**Checkpoints**: Every 50 steps = every ~50 minutes

---

## Key Takeaways

1. **RL ≠ SFT**: RL needs 3-5x more epochs for policy exploration
2. **Strategy takes time**: Epochs 3-5 are where real learning happens
3. **2 epochs = wasted effort**: You'd see format, miss strategy
4. **7 epochs = balanced**: Sufficient learning without excess
5. **Time is manageable**: ~24 hours with optimizations is practical

---

## Files Modified

✅ `grpo_local_data_peft.py`:
- Line 226: `num_train_epochs=7` (was 2)
- Lines 337-345: Updated metadata

---

## Next: Start Training

```bash
cd /Users/alexchilton/Downloads/Current_Learning/uni/BERN/module4_transformer_grpo/expert_guy/post_training_project
python grpo_local_data_peft.py
```

**Expected duration**: 20-24 hours
**Checkpoints**: Every 50 steps
**Final output**: `output5/wordle-grpo-optimized/final_model`

---

**Decision**: ✅ **7 epochs confirmed**
**Rationale**: Minimum needed for RL strategy emergence
**Time cost**: ~24 hours (acceptable)
**Alternative**: Could do 5 minimum if time-constrained
