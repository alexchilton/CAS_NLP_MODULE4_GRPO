"""
Simple script to download google/gemma-3-4b-it model

This will download the model to your local Hugging Face cache.
You must be logged in with: huggingface-cli login
And have accepted the license at: https://huggingface.co/google/gemma-3-4b-it

Usage:
    python download_gemma.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("="*80)
print("Downloading google/gemma-3-4b-it")
print("="*80)

# Download tokenizer
print("\n1. Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
print("✓ Tokenizer downloaded successfully!")

# Download model
print("\n2. Downloading model (this may take several minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-4b-it",
    device_map="auto",
    torch_dtype=torch.float32
)
print("✓ Model downloaded successfully!")

# Quick test
print("\n3. Testing model...")
test_input = "Hello, how are you?"
inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Test input: {test_input}")
print(f"Test output: {response}")

print("\n" + "="*80)
print("SUCCESS! google/gemma-3-4b-it is ready to use")
print("="*80)
