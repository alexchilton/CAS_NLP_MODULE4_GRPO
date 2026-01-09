import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model
MODEL_PATH = "output5/wordle-grpo-rtx4070/final_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

def test_wordle(secret_word, past_guesses=[]):
    """Test the model on a Wordle game"""
    
    # Build the prompt (same format as training)
    prompt = f"""<|im_start|>system

You are playing Wordle, a word-guessing game. Your goal is to guess a secret 5-letter word.

Rules:
- The secret word is exactly 5 letters
- After each guess, you receive feedback for each letter:
  - ✓ means the letter is correct and in the right position
  - (-) means the letter is in the word but in the wrong position
  - (x) means the letter is not in the word at all

Previous guesses and feedback:
"""
    
    for guess, feedback in past_guesses:
        prompt += f"Guess: {guess}\nFeedback: {feedback}\n"
    
    prompt += "\nBased on the feedback, what is your next guess?<|im_end|>\n<|im_start|>assistant\n"
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.3,  # Low temperature for focused guessing
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# Test examples
print("=" * 60)
print("TEST 1: Secret word is HOUSE")
print("=" * 60)
response = test_wordle("HOUSE", [
    ("CRANE", "C(x) R(x) A(x) N(x) E(-)")
])
print(response)
print()

print("=" * 60)
print("TEST 2: Secret word is STARE")
print("=" * 60)
response = test_wordle("STARE", [
    ("CRANE", "C(x) R(✓) A(✓) N(x) E(✓)")
])
print(response)