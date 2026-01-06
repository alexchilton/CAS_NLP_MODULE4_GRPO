# Test Base Qwen with Structured Prompts

This directory tests the **base Qwen/Qwen2.5-3B-Instruct** model using the **structured prompt system** from wordle-rl-gemma.

## Purpose

Compare how well the base model performs with:
- ✅ **Structured state summaries** (Green/Yellow/Gray lists)
- ✅ **Visual patterns** (`_ _ A _ _`)
- ✅ **Explicit constraints** (words already guessed)
- ✅ **In-context examples** (system prompt with examples)

vs. your current symbolic feedback format (`C(x) R(x) A(-) N(x) E(-)`).

## Files

- `prompt_system.py` - Structured prompt builder (adapted from wordle-rl-gemma)
- `wordle_game.py` - Wordle game engine
- `test_base_qwen.py` - Main test script
- `README.md` - This file

## Usage

### Basic Test (20 games, low temperature)
```bash
cd test_base_model_structured_prompts
python test_base_qwen.py --games 20 --temperature 0.1
```

### Creative Test (higher temperature)
```bash
python test_base_qwen.py --games 20 --temperature 0.7
```

### Large Evaluation (like their 150 games)
```bash
python test_base_qwen.py --games 150 --temperature 0.1 --seed 42
```

### Custom Model
```bash
python test_base_qwen.py --model "path/to/your/model" --games 10
```

## Arguments

- `--model`: Model name or path (default: `Qwen/Qwen2.5-3B-Instruct`)
- `--games`: Number of games to play (default: 20)
- `--temperature`: Sampling temperature (default: 0.1)
  - 0.1 = deterministic, focused (recommended)
  - 0.7 = creative, exploratory
- `--seed`: Random seed for reproducibility (default: 42)
- `--word-list`: Path to word list CSV (default: `../five_letter_words.csv`)
- `--verbose`: Print detailed game output (default: True)

## Expected Output

The script will:
1. Load the base Qwen model
2. Play N games with structured prompts
3. Print detailed output for each game (guesses, feedback, thinking)
4. Show summary statistics (win rate, avg turns, etc.)
5. Save results to a text file

## Example Output

```
================================================================================
SUMMARY
================================================================================
Model: Qwen/Qwen2.5-3B-Instruct
Temperature: 0.1
Games played: 20
Wins: 8 (40.0%)
Losses: 10 (50.0%)
Invalid: 2 (10.0%)
Average turns to win: 4.25

Detailed results:
  🎉 Game 1: CRANE - win in 4 turns
  😞 Game 2: STARE - loss in 6 turns
  ❌ Game 3: AUDIO - invalid_format in 1 turns
  ...
```

## Key Differences from Your Current Testing

| Aspect | This Test | Your test_checkpoint.py |
|--------|-----------|------------------------|
| Prompt format | Structured state summaries | Symbolic feedback |
| Visual cues | Pattern: `_ _ A _ _` | None |
| Constraints | Explicit lists | Implicit in history |
| System prompt | Full examples | Minimal instructions |
| Consistency | Same as wordle-rl-gemma | Custom format |

## What to Look For

1. **Format accuracy**: Does the base model produce valid `<think>` and `<guess>` tags?
2. **Win rate**: What % of games does it win? (Compare to their ~20-30% reported)
3. **Strategic thinking**: Does the reasoning in `<think>` make sense?
4. **Invalid guesses**: Does it guess invalid words or repeat words?
5. **Temperature sensitivity**: How much better is temp=0.1 vs temp=0.7?

## Comparison Baseline

From wordle-rl-gemma README:
- **Base Gemma-3 4B with structured prompts + game history + temp=0.1**: ~20-30% wins
- **LoRA-trained model with same setup**: ~40-50% wins

Your base Qwen (3B) should perform similarly or slightly worse (smaller model).

## Next Steps

After running this test, you can:
1. Compare win rate to your current symbolic format
2. Decide whether to adopt structured prompts for training
3. Regenerate SFT data with new format if beneficial
4. Use as baseline for evaluating your trained models
