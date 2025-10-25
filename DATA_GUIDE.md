# Wordle GRPO Data Guide

## 📍 Where is the Data?

### 1. **Downloaded Datasets** (cached locally)
```
data/cache/
├── predibase___wordle-grpo/    # Main GRPO training dataset (76 examples)
└── predibase___wordle-sft/     # Reference SFT dataset (82 examples)
```

### 2. **Wordle Word List** (12,920 five-letter words)
```
data/wordle_word_list.csv       # Complete list of valid Wordle words
```

### 3. **Metadata**
```
data/download_metadata.json     # Download timestamp and dataset info
```

---

## 🎮 Dataset Contents

### GRPO Training Dataset
- **Total examples**: 76
- **Columns**: `prompt`, `word_list`, `past_guess_history`, `secret`
- **Format**: ChatML-style prompts with game state

**Example structure**:
```python
{
    "secret": "ALLEY",
    "past_guess_history": [
        ["CRANE", "C(x) R(x) A(-) N(x) E(-)"],
        ["SWEAT", "S(x) W(x) E(-) A(-) T(x)"]
    ],
    "word_list": "https://raw.githubusercontent.com/.../five_letter_words.csv",
    "prompt": "<|im_start|>system\nYou are playing Wordle...\n<|im_end|>"
}
```

**Secret words in dataset**: 76 unique words including:
- ABHOR, ALLEY, ALLOT, CHAOS, CRACK, GLORY, NIGHT, PITCH, STONE, TRAIL, WRITE, etc.

### Word List Statistics
- **Total words**: 12,920
- **Most common letters**: S (10.3%), E (10.3%), A (9.2%)
- **Most common starting letters**: S (1,559), C (920), B (903)
- **Range**: AAHED to ZYMIC

---

## 📊 How to View the Data

### Quick View Scripts

**1. View dataset examples**:
```bash
# Show 5 examples
python scripts/view_data.py

# Show 10 examples with all secret words
python scripts/view_data.py --num-examples 10 --show-words
```

**2. View word list**:
```bash
# Show word list with statistics
python scripts/view_data.py --word-list
```

**3. Interactive exploration (Jupyter)**:
```bash
jupyter notebook notebooks/explore_data.ipynb
```

### Programmatic Access

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("predibase/wordle-grpo", cache_dir="data/cache")
train_data = dataset['train']

# Access examples
example = train_data[0]
print(f"Secret: {example['secret']}")
print(f"History: {example['past_guess_history']}")
print(f"Prompt: {example['prompt'][:100]}...")

# Load word list
with open("data/wordle_word_list.csv", 'r') as f:
    words = [line.strip() for line in f if line.strip() and line.strip() != 'Word']
print(f"Total words: {len(words)}")
```

---

## 🎯 Data for Training

### What the Model Learns From

**Input (Prompt)**:
- Game rules and instructions
- Previous guesses and their feedback
- Available word list
- Current game state

**Expected Output**:
```xml
<think>
Reasoning about what to guess next based on feedback...
</think>
<guess>BRAIN</guess>
```

**Feedback Format**:
- `✓` = Correct letter in correct position
- `-` = Correct letter in wrong position
- `x` = Letter not in word

**Example**:
```
Guess: CRANE
Feedback: C(x) R(✓) A(✓) I(x) N(-)
Meaning: R and A are correct positions, N is in word but wrong spot, C and I not in word
```

---

## 📈 Dataset Statistics

### GRPO Dataset
- **76 examples** covering various game states
- **0-5 previous guesses** per example
- **76 unique secret words**

### Guess History Distribution
- Examples with 0 guesses (first turn): ~13%
- Examples with 1-2 guesses: ~45%
- Examples with 3-4 guesses: ~35%
- Examples with 5 guesses: ~7%

### Word List Coverage
- **12,920 valid five-letter words**
- Includes common words (CRANE, STARE) and obscure words (ZUPPA, ZURFS)
- Comprehensive coverage of English vocabulary

---

## 🔍 Useful Commands

```bash
# Download/refresh datasets
python scripts/download_data.py

# View data samples
python scripts/view_data.py --num-examples 5

# View all secret words
python scripts/view_data.py --show-words

# View word list statistics
python scripts/view_data.py --word-list

# Check what's in data directory
ls -lah data/
tree data/

# Count total words
wc -l data/wordle_word_list.csv

# Search for specific words
grep "CRANE" data/wordle_word_list.csv
```

---

## 💡 Tips

1. **Word List URL**: The dataset references a GitHub URL for the word list, but we've downloaded it locally to `data/wordle_word_list.csv` for offline use

2. **Past Guess History**: Stored as string in dataset but parsed as list of `[guess, feedback]` pairs

3. **Prompt Format**: Uses ChatML format (`<|im_start|>`, `<|im_end|>`) compatible with modern instruction-tuned models

4. **Secret Words**: The training set has 76 unique target words, providing diverse examples of different game states

5. **Exploration**: Use the Jupyter notebook (`notebooks/explore_data.ipynb`) for interactive data exploration with visualizations
