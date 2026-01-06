"""
STRICT FORMAT ENFORCEMENT - Based on original with heavy penalties for hallucination

Key changes from original:
1. Added heavy penalty for ANY text after </guess> tag
2. Stricter format validation - must be exactly <think>...</think><guess>WORD</guess>
3. No partial credit for format violations
4. Simplified reward structure focused on format compliance

Reward structure:
- Perfect format + valid word: +1.0
- Invalid word but correct format: +0.5  
- Wrong length but correct format: +0.1
- ANY text after </guess>: -2.0 (HEAVY PENALTY)
- No format match: 0.0
"""

import re
import ast
import pandas as pd


def output_format_check(prompt: str, completion: str, example: dict, training_progress: float = 0.0) -> float:
    """
    STRICT format checking with heavy penalty for text after </guess>
    
    Expected format: <think>...</think><guess>WORD</guess>
    ALLOWED: <think>...</think><guess>WORD</guess><|im_end|>
    
    Penalties:
    - Text after </guess> (excluding <|im_end|>): -2.0 (prevents hallucination)
    - Wrong length: +0.1 (minimal credit)
    - Invalid word: +0.5 (some credit for format)
    - Perfect: +1.0
    """
    reward = 0.0
    
    try:
        # Add synthetic <think> as it's already in the prompt
        completion = "<think>" + completion
        
        # STRICT: Check for exact pattern with nothing after </guess> except optional <|im_end|>
        strict_regex = (
            r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*</think>\s*"
            r"<guess>\s*([\s\S]*?)\s*</guess>(?:\s*<\|im_end\|>)?\s*$"
        )
        
        # First check if there's text after </guess> - HEAVY PENALTY
        # But allow <|im_end|> as it's part of the chat template
        if "</guess>" in completion:
            after_guess = completion.split("</guess>", 1)[1].strip()
            # Remove allowed tokens
            after_guess_cleaned = after_guess.replace("<|im_end|>", "").strip()
            if after_guess_cleaned:
                # There is actual text after </guess> - this is hallucination!
                return -2.0
        
        # Check for strict format match
        match = re.search(strict_regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 2:
            # No proper format found
            return 0.0
        
        # Extract the guess
        guess = match.groups()[1].strip()
        
        # Remove any prefixes like "guessed-word:"
        guess = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess, flags=re.IGNORECASE)
        
        # Extract only letters
        guess = re.sub(r'[^A-Za-z]', '', guess).upper()
        
        # Check length
        if len(guess) != 5:
            return 0.1  # Minimal credit for trying
        
        # Check if it's a valid word
        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values
        
        if guess in word_list_upper:
            reward = 1.0  # Perfect!
        else:
            reward = 0.5  # Invalid word but correct format
        
        return reward
        
    except Exception as e:
        return 0.0


def uses_previous_feedback(prompt: str, completion: str, example: dict) -> float:
    """
    Original feedback checking logic with minor improvements
    
    Rewards:
    - Correct position kept: +0.2
    - Valid letter in new position: +0.1
    - Exploration bonus: +0.05 per new letter
    
    Penalties:
    - Reusing wrong position: -0.2
    - Using dead letter: -0.5
    """
    reward = 0.0
    
    try:
        # Add synthetic <think>
        completion = "<think>" + completion
        
        # Extract the guess
        regex = r"<guess>\s*([\s\S]*?)\s*</guess>"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            return 0.0
        
        guess = match.groups()[0].strip()
        guess = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess, flags=re.IGNORECASE)
        guess = re.sub(r'[^A-Za-z]', '', guess).upper()
        
        if len(guess) != 5:
            return 0.0
        
        # Parse past guess history
        past_guess_history = example.get("past_guess_history", [])
        if isinstance(past_guess_history, str):
            try:
                past_guess_history = ast.literal_eval(past_guess_history)
            except:
                past_guess_history = []
        
        if len(past_guess_history) == 0:
            # First guess - small exploration bonus
            return 0.1
        
        # Track feedback
        correct_letter_to_position = {}
        valid_letter_to_position = {}
        wrong_letter_to_position = {}
        
        for past_guess, past_feedback in past_guess_history:
            past_guess = past_guess.upper()
            past_feedback_parts = past_feedback.split(" ")
            
            for i, fb in enumerate(past_feedback_parts):
                if i >= len(past_guess):
                    continue
                letter = past_guess[i]
                
                if '✓' in fb or '√' in fb:
                    if letter not in correct_letter_to_position:
                        correct_letter_to_position[letter] = set()
                    correct_letter_to_position[letter].add(i)
                elif '-' in fb:
                    if letter not in valid_letter_to_position:
                        valid_letter_to_position[letter] = set()
                    valid_letter_to_position[letter].add(i)
                elif 'x' in fb or 'X' in fb:
                    if letter not in wrong_letter_to_position:
                        wrong_letter_to_position[letter] = set()
                    wrong_letter_to_position[letter].add(i)
        
        # Check current guess
        for idx, letter in enumerate(guess):
            # Positive reward if keeping confirmed correct position
            if (letter in correct_letter_to_position and idx in correct_letter_to_position[letter]):
                reward += 0.2
            # Reward if letter known to be in word is used in new position
            elif (letter in valid_letter_to_position and idx not in valid_letter_to_position[letter]):
                reward += 0.1
            # Penalize reuse of same wrong position
            elif (letter in valid_letter_to_position and idx in valid_letter_to_position[letter]):
                reward -= 0.2
            # Penalize use of dead letter
            elif letter in wrong_letter_to_position:
                reward -= 0.5
            else:
                # Reward exploration of new letters
                reward += 0.05
        
        return reward
        
    except Exception as e:
        return 0.0


def guess_value(prompt: str, completion: str, example: dict) -> float:
    """
    Original information gain calculation (simplified version)
    
    Returns normalized information gain between 0.0 and 1.0
    """
    import math
    
    def validate_guess(secret: str, guess: str, raw_feedback: bool = False) -> str:
        feedback = []
        secret_list = list(secret)
        
        # Check for correct positions
        for i, (g_char, s_char) in enumerate(zip(guess, secret)):
            if g_char == s_char:
                feedback.append(f"{g_char}(✓) ")
                secret_list[i] = None
            else:
                feedback.append(None)
        
        # Check for misplaced letters
        for i, g_char in enumerate(guess):
            if feedback[i] is None:
                if g_char in secret_list:
                    feedback[i] = f"{g_char}(-) "
                    secret_list[secret_list.index(g_char)] = None
                else:
                    feedback[i] = f"{g_char}(x) "
        
        if raw_feedback:
            return feedback
        return "".join(feedback).strip()
    
    def filter_candidates(all_candidate_words, past_guesses):
        filtered = []
        for word in all_candidate_words:
            valid = True
            for past_guess, past_feedback in past_guesses:
                candidate_feedback = validate_guess(word, past_guess)
                if candidate_feedback != past_feedback:
                    valid = False
                    break
            if valid:
                filtered.append(word)
        return filtered
    
    def compute_normalized_information_gain(all_candidate_words, past_guesses, guess):
        candidates = filter_candidates(all_candidate_words, past_guesses)
        total_candidates = len(candidates)
        
        if total_candidates == 0:
            return 0.0
        
        current_entropy = math.log2(total_candidates)
        
        feedback_groups = {}
        for word in candidates:
            feedback = validate_guess(word, guess, raw_feedback=True)
            feedback_pattern = "".join('1' if "✓" in fb else ('0' if "-" in fb else 'x') 
                                    for fb in feedback)
            feedback_groups.setdefault(feedback_pattern, []).append(word)
        
        expected_entropy = 0
        for group in feedback_groups.values():
            group_size = len(group)
            p = group_size / total_candidates
            group_entropy = math.log2(group_size) if group_size > 0 else 0
            expected_entropy += p * group_entropy
        
        expected_gain = current_entropy - expected_entropy
        normalized_expected_gain = expected_gain / current_entropy if current_entropy > 0 else 0
        
        return normalized_expected_gain
    
    reward = 0.0
    try:
        # Add synthetic <think>
        completion = "<think>" + completion
        
        # Extract the guess
        regex = r"<guess>\s*([\s\S]*?)\s*</guess>"
        match = re.search(regex, completion, re.DOTALL)
        if match is None or len(match.groups()) != 1:
            return 0.0
        
        guess = match.groups()[0].strip()
        guess = re.sub(r'^(guessed-word|word|answer)\s*:\s*', '', guess, flags=re.IGNORECASE)
        guess = re.sub(r'[^A-Za-z]', '', guess).upper()
        
        if len(guess) != 5:
            return 0.0
        
        # Load word list
        word_list = pd.read_csv(str(example["word_list"]))
        word_list_upper = word_list["Word"].str.upper().values
        
        if guess not in word_list_upper:
            return 0.0
        
        # Extract past guesses
        past_guess_history = example.get("past_guess_history", [])
        if isinstance(past_guess_history, str):
            try:
                past_guess_history = ast.literal_eval(past_guess_history)
            except:
                past_guess_history = []
        
        # Compute normalized information gain
        normalized_expected_gain = compute_normalized_information_gain(
            word_list_upper,
            past_guess_history,
            guess
        )
        
        reward = normalized_expected_gain
        
    except Exception as e:
        return 0.0
    
    return reward
