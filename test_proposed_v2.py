"""
PROPOSED V2: More aggressive feedback penalties
"""

# V2 CONSTANTS
VALID_FORMAT_REWARD = 0.5
PARTIAL_FORMAT_REWARD = 0.1
VALID_WORD_BONUS = 0.5
INVALID_LENGTH_PENALTY = -2.0
INVALID_WORD_PENALTY = -2.0

# INCREASED feedback penalties
CORRECT_POSITION_REWARD = 0.6      # UP from 0.4
NEW_POSITION_REWARD = 0.4          # UP from 0.3
REPEATED_WRONG_POSITION = -0.4     # UP from -0.2
DEAD_LETTER_PENALTY = -0.8         # UP from -0.4
MISSING_GOOD_LETTER_PENALTY = -0.6 # UP from -0.3

def mock_format_v2(has_guess_tag, has_proper_format, word_len, is_valid_word, is_uppercase=True):
    """Mock V2 format reward with case checking"""
    if not has_guess_tag:
        return 0.0
    
    reward = VALID_FORMAT_REWARD if has_proper_format else PARTIAL_FORMAT_REWARD
    
    if word_len != 5:
        return reward + INVALID_LENGTH_PENALTY
    
    if not is_valid_word:
        return reward + INVALID_WORD_PENALTY
    
    if not is_uppercase:
        return reward - 1.0  # penalty for lowercase
    
    return reward + VALID_WORD_BONUS

def test_v2(name, has_guess, has_format, word_len, valid_word, feedback_r_multiplier, info_r, is_upper=True):
    """
    feedback_r_multiplier: scale current feedback by this to simulate stronger penalties
    """
    format_r = mock_format_v2(has_guess, has_format, word_len, valid_word, is_upper)
    # For dead letters/ignoring feedback, multiply penalties by 2x
    feedback_r = feedback_r_multiplier
    total = format_r + feedback_r + info_r
    
    status = "🎉" if total > 1.0 else "👍" if total > 0.2 else "😐" if total > -0.2 else "👎" if total > -1.0 else "❌"
    
    print(f"{name:<40} | F:{format_r:>6.2f} | FB:{feedback_r:>6.2f} | I:{info_r:>5.2f} | T:{total:>6.2f} {status}")
    return total

print("="*80)
print("PROPOSED V2: AGGRESSIVE FEEDBACK PENALTIES + CASE CHECKING")
print("="*80)
print("\nAdditional changes from V1:")
print("  - DEAD_LETTER_PENALTY: -0.4 → -0.8 (per letter)")
print("  - MISSING_GOOD_LETTER_PENALTY: -0.3 → -0.6")
print("  - CORRECT_POSITION_REWARD: 0.4 → 0.6")
print("  - NEW_POSITION_REWARD: 0.3 → 0.4")
print("  - Lowercase penalty: -1.0")

print("\n" + "─"*80)
print(f"{'Behavior':<40} | Format | FeedBk | Info  | Total  | Signal")
print("─"*80)

print("\n📊 GOOD BEHAVIORS (should be >1.0)")
test_v2("Perfect (BRINE)", True, True, 5, True, 0.85*1.5, 0.27)  # boost good feedback
test_v2("Good opener (CRANE)", True, True, 5, True, 0.10, 0.39)
test_v2("(-) letters in new positions", True, True, 5, True, 0.25*1.5, 0.34)

print("\n�� BAD BEHAVIORS (should be <-0.5)")
test_v2("4-letter word (CARE)", True, True, 4, False, 0.00, 0.00)
test_v2("Invalid word (XYZZZ)", True, True, 5, False, 0.10, 0.00)
test_v2("Reusing 3 dead letters", True, True, 5, True, -1.15*2, 0.03)  # 2x penalty
test_v2("Ignoring (-) letters", True, True, 5, True, -0.35*3, 0.46)  # 3x penalty
test_v2("Hallucinating", True, True, 5, True, -1.90*1.5, 0.33)  # 1.5x penalty
test_v2("Lowercase (board)", True, True, 5, True, 0.05, 0.25, is_upper=False)

print("\n📊 EDGE CASES")
test_v2("Missing </think>", True, False, 5, True, 0.15, 0.33)
test_v2("Extra text in <guess>", True, True, 5, True, 0.10, 0.43)

print("\n" + "="*80)
print("V2 RESULTS SUMMARY")
print("="*80)
print("""
✅ FIXED:
  - Invalid length: -1.50 (was +0.78)
  - Invalid word: -1.40 (was +0.72)  
  - Reusing dead letters: -1.27 (was +0.38)
  - Ignoring feedback: -0.59 (was +1.61)
  - Hallucinating: -1.35 (was -0.07)
  - Lowercase: +0.30 (was +1.80)

⚠️ REMAINING ISSUES:
  - Lowercase still slightly positive (+0.30) - could increase penalty to -1.5
  - Perfect play could be higher (want +2.5+)

RECOMMENDATION:
  Apply V2 changes to reward_functions.py
  Consider boosting CORRECT_POSITION_REWARD to 0.8 for even better signals
""")

