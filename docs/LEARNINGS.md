# Wordle GRPO Training: Key Learnings

## Project Overview
Training language models to play Wordle using Group Relative Policy Optimization (GRPO) with reward shaping.

**Date:** October 2025
**Models Tested:** GPT-2, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct

---

## 1. Model Selection: Why GPT-2 Failed

### Initial Attempt: GPT-2
**Problem:** GPT-2 could not learn the XML output format required for Wordle gameplay.

**Required Format:**
```xml
<think>
reasoning about the guess
</think>
<guess>WORD</guess>
```

**What Happened:**
- GPT-2 consistently failed to generate valid XML tags
- Would output plain text or malformed tags
- Format reward stayed low (~0.1-0.3) throughout training
- Even after multiple epochs, could not reliably produce `<guess>...</guess>` structure

**Root Cause:** GPT-2 was not instruction-tuned and lacks structured output capabilities.

### Solution: Instruction-Tuned Models

**Switched to Qwen2.5-1.5B-Instruct:**
- ✅ Learned XML format immediately (100% format compliance after epoch 1)
- ✅ Consistently generates valid 5-letter words
- ✅ Properly uses `<think>` and `<guess>` tags
- ❌ But strategic learning remains limited (see Section 3)

**Key Learning:** For structured output tasks, use instruction-tuned models (e.g., models with "-Instruct", "-IT", or "-Chat" suffixes).

---

## 2. The `num_generations` Parameter

### What is `num_generations`?

In GRPO, for each training prompt, the model generates N different responses:

```python
num_generations: 2  # Generate 2 different guesses per prompt
```

**Example:**
- Prompt: "What's your next guess?"
- Generation 1: "THINK"
- Generation 2: "WORLD"
- Rewards: [0.3, 0.5]
- Advantages: [0.3 - 0.4, 0.5 - 0.4] = [-0.1, +0.1]

### Why It's Critical

GRPO learns by comparing multiple responses to the **same** prompt:

1. **Compute baseline:** Average reward across all generations
2. **Compute advantages:** Reward - baseline for each generation
3. **Update policy:** Increase probability of high-advantage responses, decrease low-advantage responses

### The Problem: num_generations = 2 is Too Low

**Current Configuration:** `num_generations: 2`

**Observed Issues:**
- 75% of training steps are **skipped** (loss = 0.0000)
- Both generations often produce similar outputs (e.g., both "THINK")
- Similar rewards → small advantages → weak learning signal
- When `sum(|advantages|) < 1e-6`, training step is skipped entirely

**Example of Skipped Step:**
```
Generation 1: "THINK" → reward = 0.3
Generation 2: "THINK" → reward = 0.3
Baseline = 0.3
Advantages = [0.0, 0.0]
Result: Training skipped (no gradient update)
```

### Recommended Values

| Setting | Speed | Learning Quality | Use Case |
|---------|-------|-----------------|----------|
| 2 | Fast | Poor (75% skipped) | Quick experiments |
| 4 | Medium | Better | Development |
| 8 | Slow | Good | Production |
| 16 | Very Slow | Best | Final training |

**Our Recommendation:** Increase to 4-8 for meaningful learning.

---

## 3. Why Training Loss is Often Zero

### The Phenomenon

From training logs:
```
Epoch 1, Batch 1: loss=0.0000, reward=0.3432
Epoch 1, Batch 2: loss=0.0000, reward=0.5000
Epoch 1, Batch 3: loss=0.0000, reward=0.5000
Epoch 1, Batch 4: loss=12.6395, reward=0.4000  # Only this one trains!
```

**Statistics from Epoch 2:**
- Total batches: 49
- Zero loss (skipped): 37 (75%)
- Non-zero loss (trained): 12 (25%)

### The Code Explanation

In `grpo_trainer.py:207-210`:

```python
# Skip if loss is invalid or all rewards are zero (no learning signal)
if torch.isnan(loss) or torch.isinf(loss) or rewards.sum().abs() < 1e-6:
    logger.warning(f"Skipping backward pass: loss={loss:.6f}, sum_rewards={rewards:.6f}")
    loss = torch.tensor(0.0, device=self.device)  # No gradient update!
```

### Why This Happens

**Step-by-step breakdown:**

1. **Generate responses:**
   ```python
   num_generations = 2
   generations = ["THINK", "THINK"]  # Often very similar!
   ```

2. **Compute rewards:**
   ```python
   rewards = [0.3, 0.3]  # Similar rewards for similar outputs
   ```

3. **Compute advantages:**
   ```python
   baseline = mean(rewards) = 0.3
   advantages = [0.3 - 0.3, 0.3 - 0.3] = [0.0, 0.0]
   ```

4. **Check learning signal:**
   ```python
   sum(|advantages|) = 0.0 < 1e-6  # Threshold!
   → Skip training step (loss = 0)
   ```

### Root Causes

1. **Low num_generations (2):** Not enough diversity in outputs
2. **Model stuck in local minimum:** Keeps generating "THINK"
3. **Weak reward signal:** Small differences between good and bad guesses
4. **Advantage normalization:** When std(advantages) is small, they normalize to ~0

### Impact

- **75% of training steps are wasted** (no parameter updates)
- Effective batch size is reduced by 4x
- Training is extremely slow (9 hours per epoch, but only 25% effective)
- Model learns very slowly or not at all

---

## 4. Format Learning vs Strategic Learning

### Current State: Checkpoint Epoch 1

**Format Compliance: ✅ 100%**
- All outputs have valid XML structure
- All guesses are exactly 5 letters
- All guesses are alphabetic and uppercase
- Model reliably produces `<think>...</think><guess>WORD</guess>`

**Strategic Performance: ❌ 0%**
- Win rate: 0/20 games (0%)
- Model repeatedly guesses "THINK" (appears 20+ times in 10 games)
- Does not adapt based on feedback
- Ignores information from previous guesses
- Shows no strategic reasoning

### Example Game

```
Secret word: PETAR
Guess 1: RAVEN   → Feedback: R(-) A(-) V(x) E(-) N(x)
Guess 2: THINK   → Feedback: T(-) H(x) I(x) N(x) K(x)
Guess 3: THINK   → Same guess again! (no learning from feedback)
Guess 4: THINK   → Same guess again!
Guess 5: THINK   → Same guess again!
Guess 6: TUMMY   → Finally different, but doesn't win
Result: LOSS
```

### Why This Happens

**Reward Function Weights:**
```yaml
format_weight: 1.0   # High weight on format
feedback_weight: 0.5  # Medium weight on using feedback
value_weight: 0.3     # Low weight on strategic value
```

**Observed Rewards:**
- Format check: 0.5-1.0 (consistently achieved)
- Feedback usage: 0.0-0.2 (rarely triggered)
- Guess value: 0.0-0.3 (rarely triggered)

**Result:** Model optimizes for format (easy) but ignores strategy (hard).

---

## 5. Model Size and Speed Tradeoffs

### Qwen2.5-1.5B-Instruct (Mac Training)

**Specs:**
- Parameters: 1.5B (1,089,536 trainable with LoRA)
- Device: MPS (Apple Silicon)
- Memory: ~3 GB

**Performance:**
- Format learning: ✅ Excellent (100% compliance)
- Training speed: ~60 sec/batch
- Epoch duration: ~9 hours (76 batches)
- Strategic learning: ❌ Poor (0% win rate)

### Qwen2.5-3B-Instruct (Production Server)

**Specs:**
- Parameters: 3B (2x larger)
- Device: GPU
- Memory: ~6 GB (estimated)

**Expected Performance:**
- Format learning: ✅ Should be excellent
- Training speed: Potentially slower (larger model)
- Strategic learning: ❓ Unknown, but larger model may have better reasoning

**Current Status:** Training in progress on production server

---

## 6. Why Training is So Slow

### Time Breakdown (Per Batch)

**With num_generations = 2:**
- Generate 2 completions: ~40 sec
- Compute rewards: ~5 sec
- Compute log probabilities: ~10 sec
- Backward pass + update: ~5 sec
- **Total: ~60 sec/batch**

**Projected with num_generations = 8:**
- Generate 8 completions: ~160 sec (4x slower)
- Other steps: ~20 sec
- **Total: ~180 sec/batch** (3x slower than current)

### Epoch Duration Estimates

| num_generations | Sec/Batch | Batches | Epoch Time |
|----------------|-----------|---------|------------|
| 2 (current) | 60 | 76 | 9 hours |
| 4 | 120 | 76 | 18 hours |
| 8 | 180 | 76 | 27 hours |

**But remember:** With num_generations=2, 75% of batches are skipped!
- Effective training: 9 hours × 25% = 2.25 hours of actual learning
- With num_generations=8: 27 hours × ~80% = 21.6 hours of actual learning
- **10x more effective learning time**

---

## 7. Reward Function Analysis

### Three Components

```python
CombinedReward(
    format_weight=1.0,      # Is output valid XML with 5-letter word?
    feedback_weight=0.5,    # Does guess use previous feedback?
    value_weight=0.3,       # Is guess strategically valuable?
    word_list_path="data/wordle_word_list.csv"
)
```

### Format Reward (output_format_check)

**Purpose:** Ensure model outputs valid guesses

**Scoring:**
- 1.0: Perfect format + valid word in word list
- 0.5: Valid format + 5 letters, but not in word list
- 0.1-0.4: Partial credit for having some tags
- 0.0: Complete failure

**Status:** ✅ Working well (100% of outputs get 0.5-1.0)

### Feedback Reward (uses_previous_feedback)

**Purpose:** Encourage model to use information from previous guesses

**Scoring:**
- +0.2: Reuses letter in confirmed correct position
- +0.1: Uses letter known to be in word (new position)
- +0.05: Uses new letter (exploration)
- -0.2: Reuses letter in same wrong position
- -0.5: Uses letter known to be absent

**Status:** ❌ Not working (model ignores feedback, keeps guessing "THINK")

### Value Reward (guess_value)

**Purpose:** Reward guesses that reduce uncertainty about the secret word

**Scoring:**
- Uses information theory (entropy reduction)
- 1.0: Maximum information gain
- 0.0: No information gain

**Status:** ❌ Not working (rewards stay around 0.0-0.3)

### Observed Reward Distribution

From Epoch 2 logs:
```
Mean reward: 0.36
Range: -0.05 to 0.70
Negative rewards: 2/49 (4%)
High rewards (>0.6): 5/49 (10%)
```

**Interpretation:**
- Most rewards cluster around 0.3-0.5 (format + weak feedback/value)
- Very few high rewards (good strategic guesses)
- Very few negative rewards (model avoiding obviously bad guesses)
- **Not enough variance for effective GRPO learning**

---

## 8. Configuration Used

### Qwen 1.5B Config (Mac Development)

```yaml
training:
  batch_size: 1
  num_generations: 2              # TOO LOW - should be 4-8
  max_samples: 100
  gradient_accumulation_steps: 4
  epochs: 3
  learning_rate: 5e-6

model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  use_quantization: false
  lora_rank: 8
  lora_alpha: 16

data:
  dataset_name: "predibase/wordle-grpo"
  train_split: "train[:100]"
  word_list_path: "data/wordle_word_list.csv"  # Required for rewards!
```

### Production Config (3B Model)

```yaml
training:
  batch_size: 2
  num_generations: 8              # Increased for better learning
  max_samples: -1                 # Full dataset
  gradient_accumulation_steps: 16
  epochs: 3
  learning_rate: 1e-5

model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  use_quantization: true
  lora_rank: 32
  lora_alpha: 64
```

---

## 9. Key Technical Issues Encountered

### Issue 1: Word List Not Loading

**Error:** "Could not load word list from any source"

**Root Cause:**
- `prod_config.yaml` was missing `word_list_path` field
- Without word list, format reward could not validate guesses
- Training continued but with degraded reward signal

**Fix:** Added to all configs:
```yaml
data:
  word_list_path: "data/wordle_word_list.csv"
```

### Issue 2: Evaluation Stuck on Baseline

**Symptom:** Baseline evaluation hung at 0% for 2+ hours

**Root Cause:** Unknown (process still running, consuming CPU)

**Workaround:** Killed evaluation and analyzed checkpoint-only results

### Issue 3: Gradient Clipping and NaN Detection

**Code safety measures in place:**
```python
# Check for NaN gradients
if has_nan_grad:
    logger.warning("NaN detected in gradients, skipping optimizer step")
    self.optimizer.zero_grad()

# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

**Status:** No NaN detected so far (training is numerically stable)

---

## 10. Summary of Results

### Checkpoint Epoch 1 Evaluation

**Configuration:**
- Model: Qwen2.5-1.5B-Instruct + LoRA (r=8)
- Training: 1 epoch on 100 samples (76 batches, 9 hours)
- Evaluation: 20 games

**Results:**

| Metric | Value |
|--------|-------|
| Win Rate | 0% (0/20) |
| Format Compliance | 100% |
| Avg Reward | 0.38 |
| Training Steps Skipped | 75% |
| Most Common Guess | "THINK" (20+ occurrences) |

**Conclusion:** Model learned **format** but not **strategy**.

---

## 11. Recommendations

### Immediate Actions

1. **Increase num_generations to 4-8**
   - Will slow down training but improve learning quality
   - Should reduce skipped training steps from 75% to <30%

2. **Add penalty for repeated guesses**
   - Current: No explicit penalty for guessing "THINK" 5 times
   - Proposed: -1.0 reward for repeating previous guess

3. **Reweight reward functions**
   ```yaml
   format_weight: 0.5    # Reduce (already learned)
   feedback_weight: 1.0  # Increase (critical for strategy)
   value_weight: 1.0     # Increase (critical for strategy)
   ```

### Medium-term Improvements

4. **Try supervised fine-tuning first**
   - Train on expert Wordle games before GRPO
   - Model learns basic strategy, then GRPO optimizes

5. **Increase training data**
   - Currently: 100 samples
   - Try: Full dataset (few thousand samples)

6. **Experiment with temperature**
   - Higher temperature during generation → more diverse outputs
   - More diversity → stronger learning signal

### Long-term Considerations

7. **Compare model sizes**
   - Is 3B significantly better than 1.5B for this task?
   - Does extra capacity translate to better strategic reasoning?

8. **Alternative algorithms**
   - PPO (Proximal Policy Optimization)
   - DPO (Direct Preference Optimization)
   - Expert Iteration (generate → filter best → train on best)

---

## 12. Current Status

### Mac (Qwen 1.5B)
- ✅ Completed: Epoch 1
- 🔄 In Progress: Epoch 2 (33% complete)
- ⏳ Remaining: ~6 hours for epoch 2, ~9 hours for epoch 3

### Production Server (Qwen 3B)
- 🔄 In Progress: Training started
- ❓ Status: Unknown (awaiting logs)
- 🐛 Issue: Word list path was missing (fixed in config)

---

## 13. Lessons for Presentation

### What Worked
1. ✅ Instruction-tuned models can learn structured output format
2. ✅ LoRA enables efficient fine-tuning on consumer hardware
3. ✅ Reward shaping can guide model behavior
4. ✅ GRPO framework is operational and stable

### What Didn't Work
1. ❌ GPT-2 cannot learn XML format
2. ❌ num_generations=2 is too low for effective GRPO
3. ❌ Current reward weights favor format over strategy
4. ❌ Model gets stuck in local minimum (repeating "THINK")

### Key Insights
1. 🔑 GRPO requires high num_generations for diversity
2. 🔑 75% of training steps can be wasted with poor configuration
3. 🔑 Format learning ≠ Strategic learning
4. 🔑 Reward engineering is critical and difficult
5. 🔑 Training time is a major bottleneck (9 hours/epoch)

### Future Directions
1. 🎯 Increase num_generations to 4-8
2. 🎯 Add explicit penalties for repeated guesses
3. 🎯 Consider supervised pre-training before GRPO
4. 🎯 Evaluate 3B model performance vs 1.5B
5. 🎯 Explore alternative RL algorithms (PPO, DPO)

---

## 14. Reward Weight Rebalancing and Curriculum Learning

### Current Weight Distribution Problem

**Current Configuration:**
```yaml
CombinedReward(
    format_weight=1.0,     # 55% of total possible reward
    feedback_weight=0.5,   # 28% of total possible reward
    value_weight=0.3       # 17% of total possible reward
)
```

**Total Possible Reward:**
- Format: 1.0 × 1.0 = 1.0
- Feedback: 0.5 × 1.0 = 0.5
- Value: 0.3 × 1.0 = 0.3
- **Sum: 1.8**

**Weight Distribution:**
- Format: 1.0 / 1.8 = **55%**
- Feedback: 0.5 / 1.8 = **28%**
- Value: 0.3 / 1.8 = **17%**

### Why This Is Problematic

The model has learned to optimize for format (easy, 100% success rate) while ignoring strategy (hard, requires reasoning):

**Example: Current Behavior**
```
Guess: "THINK"
- Format reward: 1.0 (perfect XML + valid word)
- Feedback reward: -0.5 (uses known-absent letters)
- Value reward: 0.1 (low information gain)
- Total: 1.0 - 0.25 + 0.03 = 0.78
```

**If it tried strategic guess:**
```
Guess: "SPARE"
- Format reward: 1.0 (perfect XML + valid word)
- Feedback reward: 0.2 (uses previous feedback)
- Value reward: 0.8 (high information gain)
- Total: 1.0 + 0.1 + 0.24 = 1.34
```

**Comparison:**
- "THINK": 0.78 total reward
- "SPARE": 1.34 total reward
- Difference: **+0.56** (72% improvement)

But with `num_generations=2`, both outputs are often "THINK" → no advantage difference → no learning!

### Proposed Rebalancing

**Phase 1 (Epoch 0): Format Learning**
```yaml
format_weight: 1.0   # High (model needs to learn XML)
feedback_weight: 0.5
value_weight: 0.3
```
Use this until format compliance reaches 90%+.

**Phase 2 (Epochs 1-3): Strategy Learning**
```yaml
format_weight: 0.3   # Reduced (already learned)
feedback_weight: 1.0 # Increased (critical for strategy)
value_weight: 1.0    # Increased (critical for strategy)
```

### Expected Impact After Rebalancing

**Example with New Weights (Phase 2):**

```
Guess: "THINK" (bad strategic choice)
- Format: 0.3 × 1.0 = 0.30
- Feedback: 1.0 × (-0.5) = -0.50
- Value: 1.0 × 0.1 = 0.10
- Total: 0.30 - 0.50 + 0.10 = -0.10
```

```
Guess: "SPARE" (good strategic choice)
- Format: 0.3 × 1.0 = 0.30
- Feedback: 1.0 × 0.2 = 0.20
- Value: 1.0 × 0.8 = 0.80
- Total: 0.30 + 0.20 + 0.80 = 1.30
```

**New Comparison:**
- "THINK": -0.10 (negative reward!)
- "SPARE": +1.30 (high reward)
- Difference: **+1.40** (stronger learning signal than +0.56)

### Combined with num_generations Increase

**Current State (num_generations=2, old weights):**
- Generation 1: "THINK" → reward = 0.78
- Generation 2: "THINK" → reward = 0.78
- Baseline: 0.78
- Advantages: [0.0, 0.0]
- **Result: Training skipped**

**After Rebalancing + num_generations=8:**
- Generation 1: "THINK" → reward = -0.10
- Generation 2: "SPARE" → reward = 1.30
- Generation 3: "WORLD" → reward = 0.60
- Generation 4: "THINK" → reward = -0.10
- Generation 5: "RAISE" → reward = 1.10
- Generation 6: "STORE" → reward = 0.85
- Generation 7: "THINK" → reward = -0.10
- Generation 8: "CRANE" → reward = 1.25
- Baseline: 0.60
- Advantages: [-0.70, +0.70, 0.0, -0.70, +0.50, +0.25, -0.70, +0.65]
- **Result: Strong learning signal! Training proceeds**

### Implementation Strategy

**Option 1: Manual Phase Switch**
1. Train epoch 0 with high format weight
2. After checkpoint, evaluate format compliance
3. If >90%, switch to strategy weights for remaining epochs

**Option 2: Curriculum Learning Code**
```python
def get_reward_weights(epoch, format_compliance_rate):
    if format_compliance_rate < 0.9:
        # Still learning format
        return {"format": 1.0, "feedback": 0.5, "value": 0.3}
    else:
        # Focus on strategy
        return {"format": 0.3, "feedback": 1.0, "value": 1.0}
```

**Option 3: Gradual Transition**
```python
def get_reward_weights(epoch):
    # Linearly interpolate from format-focused to strategy-focused
    alpha = min(epoch / 3.0, 1.0)  # 0 to 1 over 3 epochs
    format_weight = 1.0 - 0.7 * alpha    # 1.0 → 0.3
    feedback_weight = 0.5 + 0.5 * alpha  # 0.5 → 1.0
    value_weight = 0.3 + 0.7 * alpha     # 0.3 → 1.0
    return {"format": format_weight, "feedback": feedback_weight, "value": value_weight}
```

### Alternative Learning Strategy: Phase-Based Training

**Key Insight:** Since format is learned in epoch 1 (100% compliance), we should **change the objective** for subsequent epochs.

**Phase 1 (Epoch 0-1): Format Learning**
- **Goal:** Teach model to output valid XML with 5-letter words
- **Weights:** format=1.0, feedback=0.5, value=0.3
- **Success Metric:** 90%+ format compliance
- **Status for our model:** ✅ ACHIEVED (100% compliance)

**Phase 2 (Epochs 2-3): Strategy Learning**
- **Goal:** Teach model to make smart guesses using feedback
- **Weights:** format=0.3, feedback=1.0, value=1.0
- **Success Metric:** Win rate >20%, reduced repetition
- **Status for our model:** ⏭️ READY TO START

**Why This Works:**

1. **Epoch 1 taught the "easy" skill** (format)
   - Model can now reliably produce `<guess>XXXXX</guess>`
   - No need to keep rewarding this heavily

2. **Epochs 2-3 should teach the "hard" skill** (strategy)
   - By reducing format weight, we force model to optimize elsewhere
   - High feedback/value weights create pressure to play strategically
   - Negative rewards for "THINK" (uses absent letters) push exploration

3. **Prevents wasted training time**
   - Don't spend epochs 2-3 re-learning what epoch 1 already mastered
   - Focus computational budget on the unsolved problem

**Practical Implementation:**

For checkpoint epoch 1 (our current state):
```yaml
# File: configs/qwen_config_phase2.yaml
reward_weights:
  format_weight: 0.3    # Down from 1.0 (already learned)
  feedback_weight: 1.0  # Up from 0.5 (needs learning)
  value_weight: 1.0     # Up from 0.3 (needs learning)

training:
  num_generations: 8    # Up from 2 (better diversity)
  epochs: 3
  # Start from checkpoint 1, train for 2 more epochs
```

**Analogy:**
- Like teaching a child to write essays
- **Phase 1:** Learn grammar and spelling (format)
- **Phase 2:** Learn argumentation and style (strategy)
- You don't keep testing spelling in Phase 2 - it's already mastered!

### Recommendation

For current checkpoint (100% format compliance achieved):
1. ✅ Switch to Phase 2 weights immediately
2. ✅ Increase num_generations to 4-8
3. ✅ Add explicit penalty for repeated guesses (-1.0)
4. ✅ Restart training from checkpoint 1 with new configuration

**Expected Outcome:**
- Stronger learning signal (1.40 advantage difference vs 0.56)
- More training steps executed (80% vs 25%)
- Model forced to explore strategic options (negative rewards for "THINK")
- Faster convergence to strategic play

### Key Insight for Presentation

**"Reward engineering is a balancing act"**

- Too much emphasis on easy objectives → model ignores hard objectives
- Solution: **Curriculum learning** - adjust weights based on training phase
- Like teaching: master basics first (format), then focus on advanced skills (strategy)
- Critical to monitor which objectives are already achieved and rebalance accordingly

---

**Document Status:** Living document, updated October 26, 2025
**Next Update:** After epoch 2 completes and production results available
