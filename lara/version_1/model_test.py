import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import random
from prompt_templates import create_wordle_prompt
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Use the absolute path
RELATIVE_PATH = "~/nlp_m4_grpo/output_4090/wordle-grpo/final_model"
MODEL_PATH = os.path.expanduser(RELATIVE_PATH)

# 2. Verify the directory exists
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(f"Could not find directory: {MODEL_PATH}")

print(f"Loading tokenizer from local path: {MODEL_PATH}")
WORD_LIST_PATH = "five_letter_words.csv"
NUM_GAMES = 10

def get_feedback(guess, secret):
    res = []
    guess = (guess + "     ")[:5]
    for i in range(5):
        if guess[i] == secret[i]: res.append(f"{guess[i]}(✓)")
        elif guess[i] in secret: res.append(f"{guess[i]}(-)")
        else: res.append(f"{guess[i]}(x)")
    return " ".join(res)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Laptop usually needs float16 if bfloat16 isn't supported by the GPU driver
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to("cuda")
words = [str(w).upper() for w in pd.read_csv(WORD_LIST_PATH).iloc[:,0].tolist() if len(str(w))==5]

random.seed(42) # Same seed so words match the cluster test
test_words = random.sample(words, NUM_GAMES)

for word in test_words:
    history = []
    past_guesses = []  # To store (guess, feedback) tuples
    success = False
    
    print(f"\nTARGET: {word}")
    
    for i in range(1, 7):
        # 2. Use the new function to build the prompt dynamically
        # It handles the System prompt, Few-shot examples, and History all in one
        prompt = create_wordle_prompt(
            past_guesses=past_guesses,
            use_few_shot=True,
            num_examples=2
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # 3. Generate response
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        # --- FIX STARTS HERE ---
        # Slice 'out' to only include tokens AFTER the input prompt
        prompt_length = inputs.input_ids.shape[1]
        new_tokens = out[0][prompt_length:]
        
        # Decode only the newly generated tokens
        resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Since your create_wordle_prompt ends with "<think>", 
        # the model's 'new_tokens' start immediately after it.
        # We prepend it back so the regex/print looks correct.
        full_resp = "<think>" + resp
        # --- FIX ENDS HERE ---

        print(f"\n--- RAW MODEL OUTPUT (Step {i}) ---\n{full_resp}\n{'-'*40}")
        
        # 4. Extraction logic (Using the filtered response)
        # Use full_resp here to ensure tags are present
        guess_match = re.search(r"<guess>(.*?)</guess>", full_resp, re.IGNORECASE)
        guess = guess_match.group(1).strip().upper() if guess_match else "ERROR"
        
        # Sanitize
        guess = re.sub(r'[^A-Z]', '', guess)[:5]

        feedback = get_feedback(guess, word)

        print(f"Step {i}: {guess} -> {feedback}")
        
        # Update past_guesses for the next turn in the loop
        past_guesses.append((guess, feedback))
        
        if guess == word:
            success = True; break

    print("RESULT: " + ("WIN" if success else "LOSE"))