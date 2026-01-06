#!/usr/bin/env python
"""
Test script to compare Gemma-3-4b model performance on Wordle
Tests:
1. Base model (google/gemma-3-4b-it) - PRE-TRAINING
2. SFT trained model (with LoRA adapters from output_sft/wordle-sft-peft/final_model)
3. GRPO trained model (with LoRA adapters from output5/wordle-grpo-optimized/checkpoint-120)

Usage:
    python test_gemma3_4b_comparison.py base     # Test base model only
    python test_gemma3_4b_comparison.py sft      # Test SFT model only
    python test_gemma3_4b_comparison.py grpo     # Test GRPO checkpoint-120 only
    python test_gemma3_4b_comparison.py both     # Test all models (default)
"""

import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List
from datetime import datetime

from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from logger_setup import logger
import torch

# ----------------------#
# CONFIGURATION         #
# ----------------------#

BASE_MODEL_NAME = "google/gemma-3-4b-it"
SFT_ADAPTER_PATH = "output_sft/wordle-sft-peft/final_model"
GRPO_ADAPTER_PATH = "output5/wordle-grpo-optimized/checkpoint-120"
TEST_WORDS = ["CRANE", "AUDIO", "STARE", "PLUMB", "FROST"]

# ----------------------#
# GAME STRUCTURES       #
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
# PROMPT SYSTEM         #
# ----------------------#

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

### Example:
Secret Word: PLUMB

Guess 1: CLIMB → Feedback: C(x) L(✓) I(x) M(✓) B(✓)
  Analysis: L is correct at position 2, M at position 4, B at position 5
  C and I are not in the word at all

Guess 2: PLUMB → Feedback: P(✓) L(✓) U(✓) M(✓) B(✓)
  SUCCESS!

### Response Format:
Think through the problem and feedback step by step. Make sure to
first add your step by step thought process within <think> </think>
tags. Then, return your guessed word in the following format:
<guess>WORD</guess>

### CRITICAL OUTPUT FORMAT REQUIREMENTS:
⚠️ YOU MUST FOLLOW THIS EXACT FORMAT - NO EXCEPTIONS ⚠️

1. Start with <think> tags containing your reasoning
2. End with EXACTLY this format: <guess>WORD</guess>
   - Use ONLY the word itself between the tags
   - NO spaces inside the tags: <guess>CRANE</guess> ✓
   - NOT like this: <guess> CRANE </guess> ✗
   - NOT like this: <guess>I think CRANE</guess> ✗
   - JUST THE 5-LETTER WORD: <guess>CRANE</guess>

3. After </guess>, STOP IMMEDIATELY. Do not generate any additional text.

FAILURE TO FOLLOW THIS FORMAT WILL RESULT IN AN INVALID GUESS.

### IMPORTANT:
- ONLY use information from the feedback you actually received in THIS game
- If you have no past feedback, start fresh with common letters
"""

def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    """Generate user prompt with game state"""
    prompt = "Make a new 5-letter word guess."

    if past_guesses:
        prompt += "\n\nHere is the feedback from your previous guesses:"
        
        confirmed_positions = {}
        wrong_positions = {}
        dead_letters = set()

        for i, guess_obj in enumerate(past_guesses):
            prompt += f"\nGuess {i+1}: {guess_obj}"
            
            for pos, (letter, fb) in enumerate(zip(guess_obj.guess, guess_obj.feedback)):
                if fb == LetterFeedback.CORRECT:
                    confirmed_positions[pos] = letter
                elif fb == LetterFeedback.WRONG_POS:
                    if letter not in wrong_positions:
                        wrong_positions[letter] = set()
                    wrong_positions[letter].add(pos)
                elif fb == LetterFeedback.WRONG_LETTER:
                    dead_letters.add(letter)

        if confirmed_positions or wrong_positions or dead_letters:
            prompt += "\n\n### FEEDBACK SUMMARY:"
            
            if confirmed_positions:
                prompt += "\n✓ Confirmed positions (KEEP these):"
                for pos, letter in sorted(confirmed_positions.items()):
                    prompt += f"\n  Position {pos+1}: {letter}"
            
            if wrong_positions:
                prompt += "\n- Letters in word but WRONG position:"
                for letter, positions in sorted(wrong_positions.items()):
                    pos_list = ", ".join([str(p+1) for p in sorted(positions)])
                    prompt += f"\n  {letter}: NOT at position(s) {pos_list}"
            
            if dead_letters:
                prompt += "\nx Dead letters (NEVER use):"
                prompt += f"\n  {', '.join(sorted(dead_letters))}"

    return prompt

def render_prompt(past_guesses: List[GuessWithFeedback]):
    return SYSTEM_PROMPT + "\n" + render_user_prompt(past_guesses) + "\nLet me solve this step by step.\n<think>"

# ----------------------#
# GAME LOGIC            #
# ----------------------#

def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    """Calculate Wordle feedback"""
    guess = guess.upper()
    secret = secret_word.upper()
    feedback = [None] * 5
    secret_counts = {}

    for letter in secret:
        secret_counts[letter] = secret_counts.get(letter, 0) + 1

    # First pass: correct positions
    for i, (g_letter, s_letter) in enumerate(zip(guess, secret)):
        if g_letter == s_letter:
            feedback[i] = LetterFeedback.CORRECT
            secret_counts[g_letter] -= 1

    # Second pass: wrong positions
    for i, g_letter in enumerate(guess):
        if feedback[i] is None:
            if g_letter in secret_counts and secret_counts[g_letter] > 0:
                feedback[i] = LetterFeedback.WRONG_POS
                secret_counts[g_letter] -= 1
            else:
                feedback[i] = LetterFeedback.WRONG_LETTER

    return feedback

def generate_guess(generator, prompt: str) -> str:
    """Generate model output with real-time display"""
    print("\n[MODEL THINKING...]", flush=True)
    outputs = generator(prompt, max_new_tokens=2048, do_sample=True, temperature=0.8)
    completion = outputs[0]["generated_text"][len(prompt):]
    print(completion, flush=True)
    return completion

def next_turn(generator, past_guesses: List[GuessWithFeedback], secret_word: str):
    """Execute one turn of the game"""
    prompt = render_prompt(past_guesses)
    completion = generate_guess(generator, prompt)
    
    match = re.search(r"<guess>\s*(.*?)\s*</guess>", completion, re.DOTALL)
    if not match:
        print("⚠️  Warning: Model did not return a valid guess format", flush=True)
        return False
    
    guess = match.group(1).strip().upper()
    print(f"\n🎯 Model guessed: {guess}", flush=True)
    
    if len(guess) != 5:
        print(f"⚠️  Warning: Invalid guess length: {len(guess)} (expected 5)", flush=True)
        return False

    feedback = get_feedback(guess, secret_word)
    past_guesses.append(GuessWithFeedback(guess, feedback))
    
    print("\n" + "="*80)
    print("GAME STATE:")
    for i, past_guess in enumerate(past_guesses, 1):
        print(f"  {i}. {past_guess}")
    print("="*80, flush=True)
    
    if guess == secret_word:
        print("\n🎉 SUCCESS! 🎉\n", flush=True)
    elif len(past_guesses) >= 6:
        print(f"\n❌ GAME OVER - Secret was: {secret_word} ❌\n", flush=True)
    
    return True

def play_game(generator, secret_word: str):
    """Play one complete game"""
    print(f"\n{'='*80}")
    print(f"🎮 STARTING GAME - Secret word: {secret_word}")
    print(f"{'='*80}\n", flush=True)
    
    past_guesses = []
    turn = 1
    
    while turn <= 6:
        print(f"\n--- TURN {turn}/6 ---", flush=True)
        valid_guess = next_turn(generator, past_guesses, secret_word)
        
        if not valid_guess:
            print(f"❌ Invalid guess on turn {turn}", flush=True)
        
        if past_guesses and past_guesses[-1].guess == secret_word:
            win = True
            break
        
        turn += 1
    else:
        win = False
    
    return win, len(past_guesses), past_guesses

# ----------------------#
# MODEL LOADER          #
# ----------------------#

def load_model(model_type="base"):
    """Load base, SFT, or GRPO model"""
    load_dotenv()

    model_name_map = {
        "base": "BASE",
        "sft": "SFT TRAINED",
        "grpo": "GRPO CHECKPOINT-120"
    }

    display_name = model_name_map.get(model_type, "BASE")
    print(f"\n{'='*80}")
    print(f"📦 LOADING {display_name} MODEL")
    print(f"{'='*80}", flush=True)

    print(f"Loading tokenizer: {BASE_MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    print(f"Loading base model: {BASE_MODEL_NAME}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="mps",
        dtype=torch.float32
    )

    if model_type == "sft":
        print(f"Loading SFT LoRA adapters from: {SFT_ADAPTER_PATH}...", flush=True)
        model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
        print("✅ SFT model with LoRA adapters loaded!", flush=True)
    elif model_type == "grpo":
        print(f"Loading GRPO LoRA adapters from: {GRPO_ADAPTER_PATH}...", flush=True)
        model = PeftModel.from_pretrained(base_model, GRPO_ADAPTER_PATH)
        print("✅ GRPO model with LoRA adapters loaded!", flush=True)
    else:
        model = base_model
        print("✅ Base model loaded!", flush=True)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Creating text generation pipeline...", flush=True)
    # Model already on MPS via device_map, so don't specify device in pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print(f"✅ Pipeline ready!\n", flush=True)

    return generator, display_name

# ----------------------#
# MAIN TEST RUNNER      #
# ----------------------#

def run_test(model_type="base"):
    """Run test on one model configuration"""
    generator, model_name = load_model(model_type)
    
    print(f"\n{'='*80}")
    print(f"🧪 TESTING {model_name} MODEL ON WORDLE")
    print(f"{'='*80}\n", flush=True)

    results = []
    for word in TEST_WORDS:
        win, turns, guesses = play_game(generator, word)
        results.append({"word": word, "win": win, "turns": turns})

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULTS - {model_name} MODEL")
    print(f"{'='*80}")
    wins = sum(1 for r in results if r["win"])
    avg_turns = sum(r["turns"] for r in results) / len(results) if results else 0
    print(f"Total games: {len(results)}")
    print(f"Wins: {wins}/{len(results)} ({100*wins/len(results):.1f}%)")
    print(f"Average turns: {avg_turns:.2f}")
    print(f"\nDetailed results:")
    for r in results:
        status = "✅ WIN" if r["win"] else "❌ LOSS"
        print(f"  {r['word']}: {status} in {r['turns']} turns")
    print(f"{'='*80}\n", flush=True)

    return results, model_name

# ----------------------#
# MAIN                  #
# ----------------------#

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "grpo"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'#'*80}")
    print(f"# GEMMA-3-4B WORDLE PERFORMANCE COMPARISON")
    print(f"# Timestamp: {timestamp}")
    print(f"# Mode: {mode}")
    print(f"{'#'*80}\n", flush=True)

    all_results = {}

    if mode in ["base", "both"]:
        results, model_name = run_test("base")
        all_results["base"] = (results, model_name)

        # Save base results
        with open(f"results_gemma3_4b_base_{timestamp}.txt", "w") as f:
            f.write(f"BASE MODEL RESULTS - {timestamp}\n")
            f.write("="*80 + "\n")
            for r in results:
                f.write(f"{r['word']}: {'WIN' if r['win'] else 'LOSS'} in {r['turns']} turns\n")
        print(f"💾 Saved: results_gemma3_4b_base_{timestamp}.txt\n", flush=True)

    if mode in ["sft", "both"]:
        results, model_name = run_test("sft")
        all_results["sft"] = (results, model_name)

        # Save SFT results
        with open(f"results_gemma3_4b_sft_{timestamp}.txt", "w") as f:
            f.write(f"SFT TRAINED MODEL RESULTS - {timestamp}\n")
            f.write("="*80 + "\n")
            for r in results:
                f.write(f"{r['word']}: {'WIN' if r['win'] else 'LOSS'} in {r['turns']} turns\n")
        print(f"💾 Saved: results_gemma3_4b_sft_{timestamp}.txt\n", flush=True)

    if mode in ["grpo", "both"]:
        results, model_name = run_test("grpo")
        all_results["grpo"] = (results, model_name)

        # Save GRPO results
        with open(f"results_gemma3_4b_grpo_{timestamp}.txt", "w") as f:
            f.write(f"GRPO TRAINED MODEL RESULTS (Checkpoint-120) - {timestamp}\n")
            f.write("="*80 + "\n")
            for r in results:
                f.write(f"{r['word']}: {'WIN' if r['win'] else 'LOSS'} in {r['turns']} turns\n")
        print(f"💾 Saved: results_gemma3_4b_grpo_{timestamp}.txt\n", flush=True)

    # Comparison if multiple models were tested
    if mode == "both" and len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"📈 COMPARISON SUMMARY")
        print(f"{'='*80}")

        for key, (results, model_name) in all_results.items():
            wins = sum(1 for r in results if r["win"])
            print(f"{model_name:20} {wins}/{len(results)} wins")

        print(f"{'='*80}\n", flush=True)
