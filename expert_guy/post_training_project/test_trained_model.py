import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List

from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from logger_setup import logger

# ----------------------#
# 1. ENV & MODEL SETUP  #
# ----------------------#

load_dotenv()
BASE_MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "output_sft/wordle-sft-peft/final_model"
HF_TOKEN = None

# Load model based on command line argument
import sys
USE_ADAPTERS = len(sys.argv) > 1 and sys.argv[1] == "trained"

print(f"Loading base model: {BASE_MODEL_NAME}")
logger.info(f"Loading base model: {BASE_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN, device_map="auto")

if USE_ADAPTERS:
    print(f"Loading LoRA adapters from: {ADAPTER_PATH}")
    logger.info(f"Loading LoRA adapters from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Model with LoRA adapters loaded successfully!")
    logger.info("Model with LoRA adapters loaded successfully!")
else:
    print("Using base model WITHOUT adapters")
    logger.info("Using base model WITHOUT adapters")
    model = base_model

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
logger.info(f"Loaded model ({'with' if USE_ADAPTERS else 'without'} LoRA adapters)")

# ----------------------#
# 2. PROMPT TEMPLATES   #
# ----------------------#

# FIX: Use the new improved prompt (no STORM/BRAVE contamination)
SYSTEM_PROMPT = """
You are playing Wordle, a word-guessing game.

### Game Rules:
- You have **6 tries** to guess a secret **5-letter** word.
- Each guess must be a valid **5-letter English word**.
- After each guess, you will receive feedback indicating how close
your guess was.

### Feedback Format:
Each letter in your guess will receive one of three symbols:
1. ✓ : The letter is in the word and in the CORRECT position.
2. - : The letter is in the word but in the WRONG position.
3. x : The letter is NOT in the word.

### Example (GENERIC - NOT from training data):
Secret Word: PLUMB

Guess 1: CLIMB → Feedback: C(x) L(✓) I(x) M(✓) B(✓)
  Analysis: L is correct at position 2, M at position 4, B at position 5
  C and I are not in the word at all

Guess 2: PLUMB → Feedback: P(✓) L(✓) U(✓) M(✓) B(✓)
  SUCCESS!

### EXPLICIT POSITION MASKS:
When you see feedback like "R(-)", it means:
- The letter R IS in the secret word
- BUT R is NOT in the position where you guessed it
- You MUST try R in a DIFFERENT position
- NEVER reuse the same position that gave you "(-)"

When you see feedback like "T(x)", it means:
- The letter T is NOT in the secret word AT ALL
- NEVER use T again in any position
- This is a "dead letter" - eliminate it completely

When you see feedback like "B(✓)", it means:
- The letter B IS in the secret word
- AND B IS in the CORRECT position
- ALWAYS keep B in this exact position for future guesses

### Response Format:
Think through the problem and feedback step by step. Make sure to
first add your step by step thought process within <think> </think>
tags. Then, return your guessed word in the following format:
<guess> guessed-word </guess>.

### IMPORTANT CONSTRAINTS:
- DO NOT reference any specific words from previous games
- DO NOT say "In the first guess..." when you haven't made a first guess yet
- ONLY use information from the feedback you actually received in THIS game
- If you have no past feedback, start fresh with common letters (like ADIEU, RAISE, OCEAN)
"""

# ----------------------#
# 3. DATA STRUCTURES    #
# ----------------------#

class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"

@dataclass
class GuessWithFeedback:
    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:
        feedback_str = " ".join(f"{letter}({fb.value})" for letter, fb in zip(self.guess, self.feedback))
        return f"{self.guess} → Feedback: {feedback_str}"

# ----------------------#
# 4. PROMPT FUNCTIONS   #
# ----------------------#

def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    """
    FIX: Enhanced prompt with explicit position tracking
    """
    logger.info(f"Rendering user prompt with {len(past_guesses)} past guesses.")
    prompt = "Make a new 5-letter word guess."

    if past_guesses:
        prompt += "\n\nHere is the feedback from your previous guesses in THIS CURRENT GAME:"

        # FIX: Track letters explicitly with position information
        confirmed_positions = {}  # position -> letter (✓)
        wrong_positions = {}      # letter -> set of wrong positions (-)
        dead_letters = set()      # letters not in word (x)

        for i, guess_obj in enumerate(past_guesses):
            prompt += f"\nGuess {i+1}: {guess_obj}"

            # Track letter states
            for pos, (letter, fb) in enumerate(zip(guess_obj.guess, guess_obj.feedback)):
                if fb == LetterFeedback.CORRECT:
                    confirmed_positions[pos] = letter
                elif fb == LetterFeedback.WRONG_POS:
                    if letter not in wrong_positions:
                        wrong_positions[letter] = set()
                    wrong_positions[letter].add(pos)
                elif fb == LetterFeedback.WRONG_LETTER:
                    dead_letters.add(letter)

        # FIX: Add explicit summary
        if confirmed_positions or wrong_positions or dead_letters:
            prompt += "\n\n### FEEDBACK SUMMARY:"

            if confirmed_positions:
                prompt += "\n✓ Confirmed positions (KEEP these exact positions):"
                for pos, letter in sorted(confirmed_positions.items()):
                    prompt += f"\n  Position {pos+1}: {letter}"

            if wrong_positions:
                prompt += "\n- Letters in word but WRONG position (try different positions):"
                for letter, positions in sorted(wrong_positions.items()):
                    pos_list = ", ".join([str(p+1) for p in sorted(positions)])
                    prompt += f"\n  {letter}: DO NOT use at position(s) {pos_list}"

            if dead_letters:
                prompt += "\nx Dead letters (NEVER use these again):"
                prompt += f"\n  {', '.join(sorted(dead_letters))}"

    return prompt

def render_prompt(past_guesses: List[GuessWithFeedback]):
    return SYSTEM_PROMPT + "\n" + render_user_prompt(past_guesses) + "\nLet me solve this step by step.\n<think>"

# ----------------------#
# 5. MODEL INTERACTION  #
# ----------------------#

def generate_stream(prompt: str) -> str:
    logger.info("Generating model output for prompt.")
    # FIX: Use lower temperature for more stable generation during testing
    outputs = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.3)
    completion = outputs[0]["generated_text"][len(prompt):]
    logger.info(f"Model completion: {completion.strip()[:100]}")
    print(completion)
    return completion

# ----------------------#
# 6. GAME LOGIC         #
# ----------------------#

def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    """FIX: Proper Wordle feedback logic"""
    logger.info(f"Calculating feedback for guess '{guess}' against secret '{secret_word}'")

    guess = guess.upper()
    secret = secret_word.upper()
    feedback = [None] * 5
    secret_counts = {}

    # Count letters in secret
    for letter in secret:
        secret_counts[letter] = secret_counts.get(letter, 0) + 1

    # First pass: Mark correct positions
    for i, (g_letter, s_letter) in enumerate(zip(guess, secret)):
        if g_letter == s_letter:
            feedback[i] = LetterFeedback.CORRECT
            secret_counts[g_letter] -= 1

    # Second pass: Mark wrong positions
    for i, g_letter in enumerate(guess):
        if feedback[i] is None:
            if g_letter in secret_counts and secret_counts[g_letter] > 0:
                feedback[i] = LetterFeedback.WRONG_POS
                secret_counts[g_letter] -= 1
            else:
                feedback[i] = LetterFeedback.WRONG_LETTER

    logger.info(f"Feedback: {[fb.value for fb in feedback]}")
    return feedback

def next_turn(
    past_guesses: List[GuessWithFeedback],
    secret_word: str
):
    logger.info(f"Starting next turn. Past guesses: {len(past_guesses)}")
    prompt = render_prompt(past_guesses)
    completion = generate_stream(prompt)
    match = re.search(
        r"<guess>\s*(.*?)\s*</guess>", completion, re.DOTALL
    )
    if not match:
        logger.warning("Model did not return a valid guess. Skipping this turn.")
        logger.info(f"Invalid guess output: {completion.strip()[:200]}")
        print("Warning: Model did not return a valid guess. Skipping this turn.")
        return False
    guess = match.group(1).strip().upper()
    logger.info(f"Model guessed: {guess}")

    # FIX: Validate guess length
    if len(guess) != 5:
        logger.warning(f"Invalid guess length: {len(guess)} (expected 5)")
        print(f"Warning: Invalid guess length: {len(guess)} (expected 5)")
        return False

    feedback = get_feedback(guess, secret_word)
    past_guesses.append(GuessWithFeedback(guess, feedback))
    print("\n\n")
    print(("-" * 100) + "\n")
    for past_guess in past_guesses:
        print(past_guess)
    if guess == secret_word:
        logger.info("Game won!")
        print("🎉 SUCCESS 🎉")
    elif len(past_guesses) >= 6:
        logger.info("Game lost: max turns reached.")
        print("❌ better luck next time... ❌")
    return True

# ----------------------#
# 7. EXAMPLE USAGE      #
# ----------------------#

def play_game(secret_word: str):
    logger.info(f"Starting new game with secret word: {secret_word}")
    past_guesses = []
    turn = 1
    win = False
    while turn <= 6:
        print(f"\nTurn {turn}:")
        valid_guess = next_turn(past_guesses, secret_word)
        if not valid_guess:
            logger.warning("No valid guess this turn.")
            print("No valid guess this turn.")
        if past_guesses and past_guesses[-1].guess == secret_word:
            win = True
            break
        turn += 1
    # Summary
    logger.info(f"Game ended. Win: {win}. Turns: {len(past_guesses)}")
    print("\n===== GAME SUMMARY =====")
    print(f"Secret word: {secret_word}")
    print(f"Result: {'WIN' if win else 'LOSS'} in {len(past_guesses)} turn(s)")
    print("Guesses:")
    for i, guess in enumerate(past_guesses, 1):
        print(f"  {i}: {guess}")
    return win, len(past_guesses), past_guesses

if __name__ == "__main__":
    model_type = "POST-SFT TRAINED MODEL" if USE_ADAPTERS else "BASE MODEL (PRE-TRAINING)"
    print("\n" + "="*100)
    print(f"TESTING {model_type}")
    print("="*100 + "\n")
    logger.info(f"Testing {model_type}")

    test_words = ["CRANE", "AUDIO", "STARE", "PLUMB", "FROST"]

    results = []
    for word in test_words:
        print(f"\n{'='*100}")
        print(f"Testing word: {word}")
        print(f"{'='*100}\n")
        win, turns, guesses = play_game(word)
        results.append({"word": word, "win": win, "turns": turns})

    print("\n" + "="*100)
    print("FINAL RESULTS SUMMARY")
    print("="*100)
    wins = sum(1 for r in results if r["win"])
    avg_turns = sum(r["turns"] for r in results) / len(results)
    print(f"Model: {model_type}")
    print(f"Total games: {len(results)}")
    print(f"Wins: {wins}/{len(results)} ({100*wins/len(results):.1f}%)")
    print(f"Average turns: {avg_turns:.2f}")
    for r in results:
        print(f"  {r['word']}: {'WIN' if r['win'] else 'LOSS'} in {r['turns']} turns")
