# Wordle GRPO Evaluation Performance Bottleneck Analysis

**Current Performance:** 12 minutes for 5 games (24 seconds per guess)
**Target Performance:** 1-2 minutes for 5 games (2-4 seconds per guess)
**Expected Speedup:** ~8x faster

---

## Performance Bottleneck Analysis

I've identified **4 major bottlenecks** causing your 12-minute evaluation time for just 5 games:

---

### 🔴 **Bottleneck #1: Excessive `max_new_tokens=100`**

**Location:** `src/evaluation/evaluator.py:115`

```python
completion = generate_single(
    model=self.model,
    tokenizer=self.tokenizer,
    prompt=prompt,
    max_new_tokens=100,  # ❌ TOO HIGH!
    temperature=0.7,
)
```

**Problem:**
- Your XML format `<think>reasoning</think><guess>CRANE</guess>` needs **~40-60 tokens max**
- Model generates up to 100 tokens per guess = **40-60% wasted computation**
- For 5 games × 6 guesses × 30 games = **30-60% of your 12 minutes is wasted**

**Impact:** ~4-7 minutes wasted

---

### 🔴 **Bottleneck #2: Sampling Instead of Greedy Decoding**

**Location:** `src/evaluation/evaluator.py:116` + `src/model/generation.py:175`

```python
# evaluator.py
temperature=0.7,  # ❌ USING SAMPLING!

# generation.py:175
do_sample=True,  # ❌ SAMPLING OVERHEAD!
temperature=temperature,
top_p=top_p,
```

**Problem:**
- Evaluation should use **deterministic greedy decoding** (temperature=0)
- Sampling adds overhead: computing probability distributions, random sampling, nucleus filtering
- You don't need diversity during evaluation, only during training!

**Impact:** ~20-30% slower per generation

---

### 🔴 **Bottleneck #3: No Batching - Sequential Processing**

**Location:** `src/evaluation/evaluator.py:206-209` and `evaluator.py:101-161`

```python
# Games processed sequentially
for secret_word in secret_words:
    result = self.play_game(secret_word=secret_word, verbose=False)  # ❌ ONE AT A TIME!
    results.append(result)
```

```python
# Within each game, guesses are sequential
for attempt in range(self.max_attempts):
    completion = generate_single(...)  # ❌ ONE GUESS AT A TIME!
```

**Problem:**
- **5 games × 6 guesses = 30 generations happening one at a time**
- GPU underutilized: batch_size=1 for all generations
- Could batch multiple games in parallel for massive speedup

**Impact:** **5-10x slower than necessary**

---

### 🔴 **Bottleneck #4: Configuration Mismatch**

**Potential Issue:** `num_generations` confusion

The training code uses `num_generations` from config, which might be bleeding into evaluation:

```python
# training: num_generations=8 (generates 8 diverse outputs per prompt)
# evaluation: should always use num_generations=1 (single best guess)
```

If evaluation is accidentally using `num_generations > 1`, it's generating and throwing away extra completions.

---

## 📊 **Time Breakdown Estimate**

**Current state (12 minutes for 5 games):**
```
30 total generations (5 games × 6 guesses)
12 minutes / 30 = 24 seconds per generation

Breakdown per generation:
- Wasted tokens (40 extra tokens):        ~8 sec  (33%)
- Sampling overhead:                      ~5 sec  (21%)
- Sequential processing (no batching):    ~6 sec  (25%)
- Actual model inference:                 ~5 sec  (21%)
```

**After optimizations (estimated):**
```
With max_new_tokens=60:     ~16 sec/gen
With greedy decoding:       ~12 sec/gen
With batch_size=5:          ~2-3 sec/gen
```

**Expected time: ~1-2 minutes for 5 games** (6-12x speedup)

---

## ✅ **TODO: Specific Code Changes**

### **TODO #1: Reduce `max_new_tokens`** ⭐ QUICK WIN

**File:** `src/evaluation/evaluator.py:115`

```python
# BEFORE
completion = generate_single(
    model=self.model,
    tokenizer=self.tokenizer,
    prompt=prompt,
    max_new_tokens=100,  # ❌
    temperature=0.7,
)

# AFTER
completion = generate_single(
    model=self.model,
    tokenizer=self.tokenizer,
    prompt=prompt,
    max_new_tokens=60,  # ✅ Enough for XML format
    temperature=0.0,    # ✅ Greedy decoding for eval
)
```

**Expected speedup:** 2x (12 min → 6 min)
**Difficulty:** Easy (2-line change)

---

### **TODO #2: Add Greedy Decoding Mode**

**File:** `src/model/generation.py:237-273`

Add parameter to control greedy vs sampling:

```python
def generate_single(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_greedy: bool = False,  # ✅ NEW PARAMETER
) -> str:
    """Generate a single completion for a single prompt."""

    # ✅ Use greedy decoding for evaluation
    if use_greedy or temperature == 0.0:
        temperature = 1.0  # Ignored when do_sample=False
        do_sample = False
    else:
        do_sample = True

    completions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        num_generations=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=1,
        show_progress=False,
        do_sample=do_sample,  # ✅ Pass this down
    )

    return completions[0][0]
```

**File:** `src/model/generation.py:20-119`

Update `generate_completions()` signature and `_generate_batch()` to accept `do_sample`:

```python
def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    num_generations: int = 1,
    config: Optional[Any] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 1,
    show_progress: bool = True,
    do_sample: bool = True,  # ✅ NEW PARAMETER
) -> List[List[str]]:
    # ... (pass do_sample to _generate_batch)
```

**File:** `src/model/generation.py:122-198`

Update `_generate_batch()` to use conditional generation:

```python
def _generate_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    num_generations: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    do_sample: bool = True,  # ✅ NEW PARAMETER
) -> List[List[str]]:
    # ... tokenization code ...

    # Generate with conditional sampling
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
    }

    # ✅ Only add sampling params if do_sample=True
    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        gen_kwargs["do_sample"] = False

    outputs = model.generate(**inputs, **gen_kwargs)
    # ... decoding code ...
```

**Expected additional speedup:** 1.3x (6 min → 4.5 min)
**Difficulty:** Medium (refactoring generation functions)

---

### **TODO #3: Batch Game Evaluations** ⭐ BIGGEST WIN

**File:** `src/evaluation/evaluator.py`

Add a new method for batched game evaluation:

```python
def evaluate_batched(
    self,
    num_games: int = 100,
    batch_size: int = 5,  # ✅ Process 5 games in parallel
    secret_words: Optional[List[str]] = None,
    save_transcripts: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate model by playing multiple games IN PARALLEL BATCHES.

    This is much faster than sequential evaluation for games with
    independent secret words.
    """
    logger.info(f"Starting BATCHED evaluation with {num_games} games (batch_size={batch_size})")

    if secret_words is None:
        secret_words = random.sample(self.word_list, min(num_games, len(self.word_list)))
    else:
        secret_words = [w.upper().strip() for w in secret_words[:num_games]]

    results = []
    pbar = ProgressBar(total=len(secret_words), desc="Evaluating")

    # ✅ Process games in batches
    for batch_start in range(0, len(secret_words), batch_size):
        batch_end = min(batch_start + batch_size, len(secret_words))
        batch_secrets = secret_words[batch_start:batch_end]

        # Play all games in this batch in parallel
        batch_results = self._play_games_batched(batch_secrets)
        results.extend(batch_results)

        pbar.update(len(batch_secrets))
        wins = sum(1 for r in results if r["won"])
        pbar.set_postfix({"win_rate": f"{wins / len(results):.2%}"})

    pbar.close()

    # Compute metrics
    metrics = self._compute_metrics(results)
    logger.info(f"Win rate: {metrics['win_rate']:.2%}")

    if save_transcripts and output_dir:
        self.save_transcripts(results, metrics, output_dir)

    return metrics


def _play_games_batched(self, secret_words: List[str]) -> List[Dict[str, Any]]:
    """
    Play multiple games in parallel by batching guess generation.

    All games start at the same time and progress together.
    """
    batch_size = len(secret_words)

    # Initialize game states
    game_states = [
        {
            "secret_word": secret_words[i],
            "guesses": [],
            "feedbacks": [],
            "rewards": [],
            "won": False,
            "active": True,  # Still playing
        }
        for i in range(batch_size)
    ]

    # Play all games in parallel (all at same attempt number)
    for attempt in range(self.max_attempts):
        # Collect prompts for all active games
        active_indices = [i for i, gs in enumerate(game_states) if gs["active"]]
        if not active_indices:
            break

        prompts = [
            self._create_prompt(game_states[i]["guesses"], game_states[i]["feedbacks"])
            for i in active_indices
        ]

        # ✅ BATCH GENERATE ALL GUESSES AT ONCE!
        from model.generation import generate_completions
        completions_batch = generate_completions(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            num_generations=1,  # ✅ Only 1 per prompt for eval!
            max_new_tokens=60,  # ✅ Optimized for XML
            temperature=0.0,     # ✅ Greedy decoding
            batch_size=len(prompts),  # ✅ Process all at once!
            show_progress=False,
            do_sample=False,     # ✅ Greedy mode (requires TODO #2)
        )

        # Extract guesses and update game states
        for idx, active_i in enumerate(active_indices):
            completion = completions_batch[idx][0]
            guess = self._extract_guess(completion)

            # Validate and get feedback
            if guess is None or len(guess) != 5 or guess not in self.word_set:
                guess = random.choice(self.word_list)

            secret = game_states[active_i]["secret_word"]
            feedback = validate_guess(secret, guess)

            game_states[active_i]["guesses"].append(guess)
            game_states[active_i]["feedbacks"].append(feedback)

            # Check if won
            if is_winning_guess(feedback):
                game_states[active_i]["won"] = True
                game_states[active_i]["active"] = False

        # Mark games that hit max attempts
        for gs in game_states:
            if gs["active"] and len(gs["guesses"]) >= self.max_attempts:
                gs["active"] = False

    # Convert to results format
    results = [
        {
            "secret_word": gs["secret_word"],
            "guesses": gs["guesses"],
            "feedbacks": gs["feedbacks"],
            "won": gs["won"],
            "num_guesses": len(gs["guesses"]),
            "rewards": gs["rewards"],
        }
        for gs in game_states
    ]

    return results
```

**Expected additional speedup:** 4-5x (4.5 min → ~1 min)
**Difficulty:** Hard (significant refactoring)

---

### **TODO #4: Update evaluate.py to use batched evaluation**

**File:** `scripts/evaluate.py:270-278`

```python
# BEFORE
metrics = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    word_list=word_list,
    num_games=num_games,
    config=config,
    output_dir=output_dir,
    reward_function=reward_function,
)

# AFTER - Add batch_size parameter
def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    word_list: List[str],
    num_games: int = 100,
    config: Optional[Any] = None,
    output_dir: Optional[Path] = None,
    reward_function: Optional[CombinedReward] = None,
    use_batched: bool = False,  # ✅ NEW
    batch_size: int = 5,         # ✅ NEW
) -> Dict[str, Any]:
    """Convenient function to evaluate a model."""
    max_attempts = 6
    if config is not None and hasattr(config, "evaluation"):
        max_attempts = getattr(config.evaluation, "max_attempts", 6)

    evaluator = WordleEvaluator(
        model=model,
        tokenizer=tokenizer,
        word_list=word_list,
        max_attempts=max_attempts,
        reward_function=reward_function,
    )

    # ✅ Use batched evaluation if requested
    if use_batched:
        return evaluator.evaluate_batched(
            num_games=num_games,
            batch_size=batch_size,
            save_transcripts=True,
            output_dir=output_dir,
        )
    else:
        return evaluator.evaluate(
            num_games=num_games,
            save_transcripts=True,
            output_dir=output_dir,
        )
```

Then in main evaluation call:

```python
metrics = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    word_list=word_list,
    num_games=num_games,
    config=config,
    output_dir=output_dir,
    reward_function=reward_function,
    use_batched=True,      # ✅ Enable batching
    batch_size=5,           # ✅ 5 games in parallel
)
```

**Difficulty:** Easy (just exposing the new API)

---

## 📈 **Expected Performance Improvements**

| Optimization | Current | Optimized | Speedup |
|--------------|---------|-----------|---------|
| max_new_tokens | 100 | 60 | **1.4x** |
| Sampling | temperature=0.7 | greedy (temp=0) | **1.3x** |
| Batching | batch_size=1 | batch_size=5 | **4-5x** |
| **Combined** | **12 min** | **~1.5 min** | **~8x** |

---

## 🎯 **Implementation Priority**

### Phase 1: Quick Win (5 minutes of work)
- ✅ **TODO #1**: Change `max_new_tokens=60` and `temperature=0.0` in evaluator.py
- **Expected:** 12 min → 6 min (**2x speedup**)

### Phase 2: Greedy Mode (30 minutes of work)
- ✅ **TODO #2**: Add `do_sample` parameter to generation functions
- **Expected:** 6 min → 4.5 min (**1.3x additional speedup**)

### Phase 3: Batching (2 hours of work)
- ✅ **TODO #3**: Implement `evaluate_batched()` and `_play_games_batched()`
- ✅ **TODO #4**: Update evaluate.py API
- **Expected:** 4.5 min → 1-1.5 min (**4x additional speedup**)

---

## 🚀 **Quick Start: Minimal Changes**

If you want immediate improvement without refactoring, just apply **TODO #1**:

**File:** `src/evaluation/evaluator.py:115-117`

```python
# Change these two lines:
max_new_tokens=60,    # Changed from 100
temperature=0.0,       # Changed from 0.7
```

**Result:** 12 minutes → 5-6 minutes with a **2-line change!**

---

## 📝 **Testing Checklist**

After each phase:

- [ ] Run evaluation on 5 games and measure time
- [ ] Verify win rate is consistent (greedy decoding may change results slightly)
- [ ] Check that all games complete successfully
- [ ] Verify transcripts are saved correctly
- [ ] Monitor GPU memory usage (batching uses more memory)

---

## 🐛 **Potential Issues**

### Issue 1: Memory OOM with Batching

If batch_size=5 causes OOM on your 16GB GPU:

```python
# Try smaller batch size
batch_size=3  # or 2
```

### Issue 2: Different Results with Greedy Decoding

Greedy decoding (`temperature=0.0`) may produce slightly different guesses than sampling (`temperature=0.7`). This is expected and desirable for reproducibility.

### Issue 3: Batching Changes Game Order

Batched evaluation processes games in parallel, so progress bar might show different win rates at different points. Final metrics will be identical.

---

**Document created:** 2025-10-27
**Status:** TODO (not yet implemented)
**Priority:** HIGH (8x speedup available)
