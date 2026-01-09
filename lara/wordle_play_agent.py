import os
import re
import random
import argparse
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Import your custom extraction function
from reward_functions import extract_guess_from_completion

# ----------------------#
# 1. WORD LIST & FEEDBACK (No changes needed)
# ----------------------#

def load_word_list(csv_path: str = "five_letter_words.csv"):
    """
    Load allowed 5-letter words from your CSV (column 'Word').
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Word list CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    words = []
    for w in df["Word"]:
        if isinstance(w, str):
            w = w.strip().lower()
            if len(w) == 5 and w.isalpha():
                words.append(w)
    words = sorted(set(words))
    print(f"Loaded {len(words)} 5-letter words from {csv_path}")
    return words


def compute_wordle_feedback(guess: str, secret: str) -> str:
    """
    Compute Wordle-style feedback for a guess vs secret.
    Returns e.g. 'C(x) R(✓) A(-) N(x) E(x)'.
    """
    guess = guess.lower()
    secret = secret.lower()
    assert len(guess) == 5 and len(secret) == 5

    feedback = [""] * 5
    secret_chars = list(secret)

    # First pass: correct positions (greens)
    for i in range(5):
        if guess[i] == secret[i]:
            feedback[i] = "(✓)"
            secret_chars[i] = None  # consume this letter

    # Second pass: wrong-position (yellow) vs absent (gray)
    for i in range(5):
        if feedback[i]:  # already green
            continue
        if guess[i] in secret_chars:
            feedback[i] = "(-)"
            idx = secret_chars.index(guess[i])
            secret_chars[idx] = None
        else:
            feedback[i] = "(x)"

    feedback_str = " ".join(f"{guess[i].upper()}{feedback[i]}" for i in range(5))
    return feedback_str

# ----------------------#
# 2. MATCHING THE TRAINING PROMPT
# ----------------------#
def build_wordle_prompt(history):
    # This MUST match the PROMPT_SUFFIX in your training script
    instructions = (
        "Solve Wordle. Use the feedback history to find the secret.\n"
    )
    
    if not history:
        user_part = "This is your first guess."
    else:
        lines = ["Previous guesses and feedback:"]
        for g, fb in history:
            lines.append(f"- {g.upper()} : {fb}")
        user_part = "\n".join(lines)

    # Recreating the exact suffix the model was trained on
    prompt_suffix = (
        "\n\nInstructions: Think step-by-step. "
        "1. List letters that are definitely NOT in the word. "
        "2. List letters you MUST include. "
        "3. Pick a valid 5-letter word that follows these rules. "
        "Format: <think>your reasoning</think><solution>WORD</solution>"
    )

    # Qwen-style chat formatting
    full_prompt = f"user\n{instructions}{user_part}{prompt_suffix}\nassistant\n"
    return full_prompt

def extract_action(model_output):
    """
    Improved extraction to handle tag variations and noisy model output.
    """
    # Normalize: remove markdown bolding and extra whitespace
    clean_output = model_output.replace("**", "").strip()

    # 1. Primary: Look for <guess> tags (most accurate)
    # Allows for optional spaces or newlines inside the tags
    tag_match = re.search(r"<guess>\s*([A-Za-z]{5})\s*</guess>", clean_output, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).upper()

    # 2. Secondary: Look for any 5-letter word at the very end of the output
    # Models often summarize their choice last: "My guess is HELLO"
    # We use \b to ensure it's a whole word and not part of a longer one.
    all_words = re.findall(r"\b[A-Za-z]{5}\b", clean_output)
    
    # Filter out common "meta" words that are 5 letters but not guesses
    meta_words = {"THINK", "GUESS", "VALID"}
    potential_guesses = [w.upper() for w in all_words if w.upper() not in meta_words]

    if potential_guesses:
        # Return the last valid-looking 5-letter word found
        return potential_guesses[-1]
        
    return None

# ----------------------#
# 3. UPDATED GAME LOOP (Showing Thought Process)
# ----------------------#
import random
import torch

def play_one_game(model, tokenizer, secret_word, word_list, max_steps=6, verbose=True):
    """
    Plays a single game of Wordle.
    Returns: (bool won, int steps_taken, list history)
    """
    secret_word = secret_word.upper()
    history = []
    game_won = False

    if verbose:
        print(f"SECRET WORD: {secret_word}")

    for step in range(1, max_steps + 1):
        # 1. Format the prompt (Matches the GRPO training style)
        prompt = f"Target word is 5 letters. History:\n"
        for h_guess, h_feedback in history:
            prompt += f"Guess: {h_guess} | Feedback: {h_feedback}\n"
        prompt += "\nThink and then provide your <guess>WORD</guess>."

        # 2. Inference
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=True, 
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the NEW tokens generated
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = full_text[len(prompt):].strip()

        # 3. Extract the action using the helper function
        guess = extract_action(response_text)
        
        # 4. Validate guess
        if guess is None or len(guess) != 5 or guess.lower() not in [w.lower() for w in word_list]:
            if verbose:
                print(f"--- STEP {step} ---")
                print(f"Model output invalid or word not in list. Falling back to random.")
            guess = random.choice(word_list).upper()
        else:
            if verbose:
                print(f"--- STEP {step} ---")
                print(f"MODEL GUESS: {guess}")

        # 5. Generate Feedback
        feedback_list = []
        for i in range(5):
            if guess[i] == secret_word[i]:
                feedback_list.append(f"{guess[i]}(✓)") # Green
            elif guess[i] in secret_word:
                feedback_list.append(f"{guess[i]}(-)") # Yellow
            else:
                feedback_list.append(f"{guess[i]}(x)") # Gray
        
        feedback = " ".join(feedback_list)
        if verbose:
            print(f"FEEDBACK: {feedback}")
            
        history.append((guess, feedback))

        # 6. Check Win Condition
        if guess == secret_word:
            if verbose:
                print(f"SUCCESS! Solved in {step} steps.")
            game_won = True
            return True, step, history
            
    if verbose:
        print(f"FAILED. The word was {secret_word}.")
        
    return False, max_steps, history

def play_many_games(model, tokenizer, word_list, num_games: int = 10, max_steps: int = 6):
    """
    Play multiple games with random secrets from word_list and report win rate.
    """
    wins = 0
    steps_when_won = []

    for i in range(num_games):
        secret = random.choice(word_list)
        print(f"\n##### GAME {i + 1} / {num_games} #####")
        won, steps, _ = play_one_game(model, tokenizer, secret, word_list, max_steps=max_steps, verbose=True)
        if won:
            wins += 1
            steps_when_won.append(steps)

    win_rate = wins / num_games if num_games > 0 else 0.0
    avg_steps = sum(steps_when_won) / len(steps_when_won) if steps_when_won else 0.0

    print(f"\n=== Summary over {num_games} games ===")
    print(f"Win rate: {wins}/{num_games} = {win_rate:.2%}")
    if steps_when_won:
        print(f"Average steps (wins only): {avg_steps:.2f}")

# ----------------------#
# 4. CORRECT LOADING LOGIC
# ----------------------#
def main():
    parser = argparse.ArgumentParser(description="Play Wordle with a trained agent.")
    
    # Existing arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base model name")
    
    # NEW arguments you were trying to use
    parser.add_argument("--word-list", type=str, default="five_letter_words.csv", help="Path to word list csv")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--max-steps", type=int, default=6, help="Max guesses per game")

    args = parser.parse_args()

    # Now use the new arguments in your logic:
    # Example: wordle_env = WordleEnvironment(word_list=args.word_list)
    # Example: for game in range(args.num_games): ...
    
    # 1. Load Base Model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 2. Load the LoRA Adapter
    print(f"Loading adapter from {args.checkpoint}...")
    model = PeftModel.from_pretrained(base_model, args.checkpoint)

    word_list = load_word_list("five_letter_words.csv")

    play_many_games(
        model, 
        tokenizer, 
        word_list, 
        num_games=args.num_games, 
        max_steps=args.max_steps
    )

if __name__ == "__main__":
    main()