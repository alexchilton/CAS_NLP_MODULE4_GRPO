#!/usr/bin/env python
"""
Simple single-word Wordle test for Gemma-3-4b
Usage:
    python test_gemma3_4b_single.py base CRANE      # Test base model on CRANE
    python test_gemma3_4b_single.py trained AUDIO   # Test SFT model on AUDIO
"""

import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_NAME = "google/gemma-3-4b-it"
ADAPTER_PATH = "output_sft/wordle-sft-peft/final_model"

class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"

@dataclass
class GuessWithFeedback:
    guess: str
    feedback: List[LetterFeedback]
    def __repr__(self):
        fb = " ".join(f"{l}({f.value})" for l, f in zip(self.guess, self.feedback))
        return f"{self.guess} → {fb}"

SYSTEM_PROMPT = """You are playing Wordle. Guess a 5-letter word.
After each guess, you get feedback:
✓ = correct position
- = wrong position  
x = not in word

Respond with: <think>your reasoning</think><guess>WORD</guess>"""

def render_prompt(past_guesses):
    prompt = SYSTEM_PROMPT + "\n\n"
    if past_guesses:
        prompt += "Previous guesses:\n"
        for i, g in enumerate(past_guesses, 1):
            prompt += f"{i}. {g}\n"
    prompt += "\nYour guess:\n<think>"
    return prompt

def get_feedback(guess: str, secret: str):
    guess, secret = guess.upper(), secret.upper()
    feedback = [None] * 5
    counts = {}
    for c in secret:
        counts[c] = counts.get(c, 0) + 1
    for i, (g, s) in enumerate(zip(guess, secret)):
        if g == s:
            feedback[i] = LetterFeedback.CORRECT
            counts[g] -= 1
    for i, g in enumerate(guess):
        if feedback[i] is None:
            if g in counts and counts[g] > 0:
                feedback[i] = LetterFeedback.WRONG_POS
                counts[g] -= 1
            else:
                feedback[i] = LetterFeedback.WRONG_LETTER
    return feedback

def play_game(generator, secret):
    print(f"\n{'='*60}\n🎮 SECRET WORD: {secret}\n{'='*60}\n")
    past_guesses = []
    
    for turn in range(1, 7):
        print(f"\n--- TURN {turn}/6 ---")
        prompt = render_prompt(past_guesses)
        
        print("[Generating...]")
        outputs = generator(prompt, max_new_tokens=2048, do_sample=True, temperature=0.8)
        completion = outputs[0]["generated_text"][len(prompt):]
        print(completion[:300])  # Show first 300 chars
        
        match = re.search(r"<guess>\s*(\w+)\s*</guess>", completion, re.DOTALL)
        if not match:
            print("❌ No valid guess found")
            continue
        
        guess = match.group(1).strip().upper()
        print(f"\n🎯 GUESS: {guess}")
        
        if len(guess) != 5:
            print(f"❌ Wrong length: {len(guess)}")
            continue
        
        feedback = get_feedback(guess, secret)
        past_guesses.append(GuessWithFeedback(guess, feedback))
        
        print(f"📊 FEEDBACK: {past_guesses[-1]}")
        
        if guess == secret.upper():
            print(f"\n🎉 WON in {turn} turns! 🎉\n")
            return True, turn
    
    print(f"\n❌ LOST - Answer was {secret}\n")
    return False, 6

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_gemma3_4b_single.py <base|trained> <WORD>")
        sys.exit(1)
    
    use_adapters = sys.argv[1] == "trained"
    secret_word = sys.argv[2].upper()
    
    print(f"\n{'#'*60}")
    print(f"# MODEL: {'SFT TRAINED' if use_adapters else 'BASE'}")
    print(f"# WORD: {secret_word}")
    print(f"{'#'*60}\n")
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, device_map="auto")
    
    if use_adapters:
        print(f"Loading adapters from {ADAPTER_PATH}...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        model = base_model
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print("Creating pipeline...")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    print("✅ Ready!\n")
    
    win, turns = play_game(generator, secret_word)
    
    print(f"\n{'='*60}")
    print(f"RESULT: {'WIN' if win else 'LOSS'} in {turns} turns")
    print(f"{'='*60}\n")
