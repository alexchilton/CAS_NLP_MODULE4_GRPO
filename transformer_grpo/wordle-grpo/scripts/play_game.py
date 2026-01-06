
#!/usr/bin/env python3
"""
Script to play a full game of Wordle with a given model.
"""

import argparse
import sys
from pathlib import Path
import torch
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_config
from utils.device import get_device
from model.setup import load_model_and_tokenizer
from data.wordle_game import validate_guess

def parse_guess_from_xml(xml_string: str) -> str:
    """Extracts the guess from the model's XML output."""
    match = re.search(r"<guess>\s*(\w+)\s*</guess>", xml_string, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback for non-XML output
    lines = xml_string.strip().split('\n')
    return lines[-1].strip().upper()

def play_game_and_log(model, tokenizer, device, secret_word: str, log_file):
    """Plays a full game of Wordle and logs the transcript."""
    
    system_prompt = """<|im_start|>system
You are playing Wordle, a word-guessing game. Your goal is to guess the secret 5-letter word in 6 tries.
After each guess, you will receive feedback. Use this feedback to inform your next guess.
Your response must be in the following format:
<think>Your reasoning for the guess.</think>
<guess>YOUR_GUESS</guess>
<|im_end|>"""
    
    history = []
    
    for i in range(6):
        log_file.write(f"\n--- Turn {i+1} ---\n")
        
        if i == 0:
            prompt = f"{system_prompt}\n<|im_start|>user\nMake your first 5-letter word guess.<|im_end|>\n<|im_start|>assistant\n"
        else:
            past_guesses = "\n".join(history)
            prompt = f"{system_prompt}\n{past_guesses}\n<|im_start|>user\nMake a new 5-letter word guess based on the feedback.<|im_end|>\n<|im_start|>assistant\n"

        log_file.write("--- Model Prompt ---\n")
        log_file.write(prompt + "\n")

        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, do_sample=False)
        
        full_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = full_generated_text.split("<|im_start|>assistant")[-1].strip()
        
        log_file.write("\n--- Full Model Output ---\n")
        log_file.write(assistant_response + "\n")
        
        guess = parse_guess_from_xml(assistant_response)
        log_file.write(f"\nModel's Guess: {guess}\n")
        
        if len(guess) != 5:
            feedback = "Invalid guess: must be a 5-letter word."
        else:
            feedback = validate_guess(secret_word, guess)
        
        log_file.write(f"Feedback: {feedback}\n")
        
        history.append(f"<|im_start|>assistant\n{assistant_response}<|im_im_end|>")
        history.append(f"<|im_start|>user\nFeedback for your guess '{guess}': {feedback}<|im_im_end|>")

        if guess == secret_word:
            log_file.write("\n--- Game Over: You Won! ---\n")
            return

    log_file.write("\n--- Game Over: You Lost! ---\n")

def main():
    """Main function to run game tests and log them."""
    
    device = get_device()
    secret_word = "CRANE"
    log_path = "game_log.txt"

    with open(log_path, "w") as log_file:
        # --- Test Base Model ---
        log_file.write("="*60 + "\n")
        log_file.write(" " * 20 + "Testing Base Model (Before SFT)\n")
        log_file.write("="*60 + "\n")
        base_config = load_config("configs/sft_config.yaml")
        base_model, tokenizer = load_model_and_tokenizer(base_config, device)
        play_game_and_log(base_model, tokenizer, device, secret_word, log_file)

        # --- Test Fine-Tuned Model ---
        log_file.write("\n" + "="*60 + "\n")
        log_file.write(" " * 20 + "Testing Fine-Tuned Model (After SFT)\n")
        log_file.write("="*60 + "\n")
        sft_config = load_config("configs/sft_config.yaml")
        sft_model, tokenizer = load_model_and_tokenizer(sft_config, device, model_dir="sft_model")
        play_game_and_log(sft_model, tokenizer, device, secret_word, log_file)

    print(f"Game transcripts have been saved to {log_path}")

if __name__ == "__main__":
    main()
