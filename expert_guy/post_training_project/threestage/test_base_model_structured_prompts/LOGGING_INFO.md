# Logging Information

## Two Output Files Created

### 1. **Console Output** (truncated for readability)
Shows during execution:
- Game number and secret word
- Turn number
- Prompt (Turn 1 only, first 300 chars)
- RAW OUTPUT (truncated to 400 chars)
- Extracted thinking (truncated to 150 chars)
- Guess and feedback
- Win/loss status

### 2. **Detailed Log File** (`detailed_log_temp{temp}_games{N}.txt`)
**FULL, UNTRUNCATED logging of everything:**

Per game:
```
================================================================================
Playing Wordle | Secret: CRANE
================================================================================

--- Turn 1/6 ---

FULL PROMPT:
System: You are an expert Wordle-solving AI...
User: This is the first turn. Please provide your best starting word.

FULL RAW OUTPUT:
<think>This is the first guess with no prior clues. The best strategy is to use
a word with common, distinct letters to maximize information gain. I'll choose
a word with a good mix of common vowels and consonants...</think>
<guess>SLATE</guess>

FULL THINKING:
This is the first guess with no prior clues. The best strategy is to use
a word with common, distinct letters to maximize information gain. I'll choose
a word with a good mix of common vowels and consonants...

EXTRACTED GUESS: SLATE

GUESS: SLATE
FEEDBACK: S(x) L(x) A(-) T(x) E(-)
STATUS: continue

--- Turn 2/6 ---

FULL PROMPT:
System: You are an expert Wordle-solving AI...
User: You are playing a game of Wordle. Analyze the clues and provide your next guess.
**Current Knowledge:**
*   **Correct Position (Green):** `_ _ _ _ _`
*   **Wrong Position (Yellow):** 'A', 'E'
*   **Not in Word (Gray):** L, S, T
*   **Words Already Guessed:** SLATE

Your task is to find a valid 5-letter English word that fits all the clues above.
Provide your reasoning within <think> tags, and then your final guess within <guess> tags.

FULL RAW OUTPUT:
[complete untruncated model output]

FULL THINKING:
[complete untruncated thinking]

[... continues for all turns ...]
```

### 3. **Summary Results File** (`results_base_qwen_temp{temp}_games{N}.txt`)
Statistics only:
```
Model: Qwen/Qwen2.5-3B-Instruct
Temperature: 0.1
Games: 20
Wins: 8/20 (40.0%)
Losses: 10/20 (50.0%)
Invalid: 2/20 (10.0%)
Avg turns to win: 4.25

Detailed results:
Game 1: CRANE - win in 4 turns - ['SLATE', 'POWER', 'CRANE']
Game 2: STARE - loss in 6 turns - ['SLATE', 'MOUND', 'PRICE', 'STORY', 'STARE']
...
```

## What Gets Logged in Detail

✅ **Full System Prompt** (every turn, though truncated in detailed log to save space)
✅ **Full User Prompt** (complete state summary with all clues)
✅ **Full Raw Model Output** (everything the model generated, no truncation)
✅ **Full Thinking** (complete `<think>` content, no truncation)
✅ **Extracted Guess** (the parsed word)
✅ **Feedback** (in readable format with ✓/−/x symbols)
✅ **Status** (win/loss/continue)

## Usage

Run the test:
```bash
python test_base_qwen.py --games 5 --temperature 0.1
```

This creates:
- `detailed_log_temp0.1_games5.txt` ← **FULL LOGS HERE**
- `results_base_qwen_temp0.1_games5.txt` ← Summary stats

## Why This Matters

You can now:
1. **Analyze model reasoning** - See exactly what the model is thinking
2. **Debug failures** - Understand why it made bad guesses
3. **Compare strategies** - See how reasoning changes with temperature
4. **Verify prompt format** - Ensure structured prompts are working correctly
5. **Share results** - Complete logs for analysis or debugging
