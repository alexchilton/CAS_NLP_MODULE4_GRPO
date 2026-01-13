import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIG ---
# 1. PASTE THE PATH FROM YOUR 'FIND' COMMAND HERE
ADAPTER_PATH = "/storage/homefs/ln23f031/nlp_m4_grpo/qwen2.5-3b-wordle-sft/final_adapter" 
BASE_MODEL = "Qwen/Qwen2.5-3b-Instruct"
TARGET_WORD = "ALARM" 

# --- LOAD ---
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")

if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ Fine-tuned Adapter Loaded!")
else:
    print("❌ STILL NO ADAPTER FOUND. Running base model only.")

# --- GAME ENGINE ---
def get_wordle_feedback(guess, target):
    feedback = []
    for i in range(5):
        if guess[i] == target[i]: feedback.append("Green")
        elif guess[i] in target: feedback.append("Yellow")
        else: feedback.append("Grey")
    return feedback

history = ""
target_word = "ALARM" # The secret word

for i in range(1, 7):
    # This prompt structure matches your V3 training data
    prompt = f"### Instruction:\nWordle Game:\nGuesses:\n{history}\nTarget: Find the 5-letter word.\n\n### Response:\n<think>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Updated generation settings to prevent looping
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        temperature=0.6, 
        top_p=0.95, 
        do_sample=True
    )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_response = full_text[len(prompt)-7:] # Grab the reasoning + guess
    
    print(f"\n--- ROUND {i} ---\n{new_response}")
    
    guess = input("What is the model's guess? (5 letters): ").strip().upper()
    if guess == target_word:
        print("🎉 SUCCESS! The model won.")
        break
        
    # Standard Wordle Feedback Logic
    feedback = []
    for idx, char in enumerate(guess):
        if char == target_word[idx]: feedback.append("Green")
        elif char in target_word: feedback.append("Yellow")
        else: feedback.append("Grey")
        
    history += f"{i}. {guess} - Response: {feedback}\n"

for attempt in range(1, MAX_ATTEMPTS + 1):
    prompt = f"Wordle Game:\nGuesses:\n{history}\nTarget: Find the 5-letter word.\nReasoning:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.1, do_sample=True)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the guess - usually the last word after "Guess:" or similar
    # For now, let's assume the model outputs "Guess: WORD" at the end
    print(f"\n--- ATTEMPT {attempt} ---")
    print(response[len(prompt)-10:]) # Print the new part of the reasoning
    
    # MANUAL OVERRIDE: Since models can be messy, type the guess it suggested
    current_guess = input(f"What word did the model suggest? ").strip().upper()
    
    if current_guess == TARGET_WORD:
        print(f"🎉 SUCCESS! The model solved it in {attempt} tries.")
        break
        
    feedback = get_feedback(current_guess, TARGET_WORD)
    history += f"{attempt}. {current_guess} - Response: {feedback}\n"