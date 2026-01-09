import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import random

# --- CONFIG ---
MODEL_PATH = "./output5/wordle-grpo-rtx4070/final_model" 
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
    success = False
    print(f"\nTARGET: {word}")
    for i in range(1, 7):
        prompt = f"""<|system|>
        You are a strategic Wordle solver. You use your <think> block to:
        1. List letters that are confirmed (Green) or misplaced (Yellow).
        2. List letters that are eliminated (Gray).
        3. Cross-reference new candidates against all previous feedback.
        4. Output only a valid 5-letter word in the <answer> tags.
        <|user|>
        History: {history}
        Target is a 5-letter word. Think carefully and guess.
        """
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Restricted to 128 because the 4070 model wasn't trained for long thoughts
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        resp = tokenizer.decode(out[0], skip_special_tokens=True)
        
        # Old model might not have used tags well, so we take the last 5-letter word
        guess = resp.split()[-1].strip().upper()[:5]
        
        feedback = get_feedback(guess, word)
        print(f"Step {i}: {guess} -> {feedback}")
        history.append(f"{guess}:{feedback}")
        
        if guess == word:
            success = True; break
    print("RESULT: " + ("WIN" if success else "LOSE"))