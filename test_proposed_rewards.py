"""
Test PROPOSED reward rebalancing
"""

# PROPOSED NEW CONSTANTS
VALID_FORMAT_REWARD = 0.5          # REDUCED from 1.0
PARTIAL_FORMAT_REWARD = 0.1        # REDUCED from 0.3
VALID_WORD_BONUS = 0.5             # KEEP
INVALID_LENGTH_PENALTY = -2.0      # INCREASED from -0.3
INVALID_WORD_PENALTY = -2.0        # INCREASED from -0.5

# Other penalties remain similar but we'll scale them
DEAD_LETTER_PENALTY_EACH = -0.5    # per dead letter reused
MISSING_GOOD_LETTER_PENALTY = -0.4 # per (-) letter ignored

def mock_format_reward(has_guess_tag, has_proper_format, word_len, is_valid_word):
    """Mock the proposed format reward"""
    if not has_guess_tag:
        return 0.0
    
    reward = VALID_FORMAT_REWARD if has_proper_format else PARTIAL_FORMAT_REWARD
    
    if word_len != 5:
        return reward + INVALID_LENGTH_PENALTY
    
    if not is_valid_word:
        return reward + INVALID_WORD_PENALTY
    
    return reward + VALID_WORD_BONUS

def test_proposed(name, has_guess, has_format, word_len, valid_word, feedback_r, info_r):
    """Test with proposed rewards"""
    format_r = mock_format_reward(has_guess, has_format, word_len, valid_word)
    total = format_r + feedback_r + info_r
    
    print(f"\n{name}")
    print(f"  Format: {format_r:>6.2f} | Feedback: {feedback_r:>6.2f} | Info: {info_r:>6.2f} | TOTAL: {total:>6.2f}")
    
    if total > 1.0:
        print("  🎉 STRONG POSITIVE")
    elif total > 0.2:
        print("  👍 Positive")
    elif total > -0.2:
        print("  😐 Weak")
    elif total > -1.0:
        print("  👎 Negative")
    else:
        print("  ❌ STRONG NEGATIVE")
    
    return total

print("="*70)
print("PROPOSED REWARD REBALANCING")
print("="*70)
print("\nChanges:")
print("  - VALID_FORMAT_REWARD: 1.0 → 0.5")
print("  - PARTIAL_FORMAT_REWARD: 0.3 → 0.1")
print("  - INVALID_LENGTH_PENALTY: -0.3 → -2.0")
print("  - INVALID_WORD_PENALTY: -0.5 → -2.0")

print("\n\n📊 GOOD BEHAVIORS")
print("─"*70)

test_proposed(
    "Perfect guess (BRINE example)",
    True, True, 5, True,
    feedback_r=0.85, info_r=0.27
)

test_proposed(
    "Good opener (CRANE)",
    True, True, 5, True,
    feedback_r=0.10, info_r=0.39
)

test_proposed(
    "Using (-) letters in new positions",
    True, True, 5, True,
    feedback_r=0.25, info_r=0.34
)

print("\n\n📊 BAD BEHAVIORS")
print("─"*70)

test_proposed(
    "4-letter word (CARE)",
    True, True, 4, False,  # wrong length
    feedback_r=0.00, info_r=0.00
)

test_proposed(
    "Invalid word (XYZZZ)",
    True, True, 5, False,  # not in dict
    feedback_r=0.10, info_r=0.00
)

test_proposed(
    "Reusing 3 dead letters (CRAZE)",
    True, True, 5, True,
    feedback_r=-1.15, info_r=0.03
)

test_proposed(
    "Ignoring all (-) letters (SLOTH)",
    True, True, 5, True,
    feedback_r=-0.35, info_r=0.46
)

test_proposed(
    "Hallucinating fake feedback (BOOKS)",
    True, True, 5, True,
    feedback_r=-1.90, info_r=0.33
)

test_proposed(
    "Lowercase (board) - assume still valid word",
    True, True, 5, True,
    feedback_r=0.05, info_r=0.25
)

print("\n\n📊 EDGE CASES")
print("─"*70)

test_proposed(
    "Missing </think> tag",
    True, False, 5, True,  # partial format
    feedback_r=0.15, info_r=0.33
)

test_proposed(
    "Extra text in <guess>",
    True, True, 5, True,
    feedback_r=0.10, info_r=0.43
)

print("\n\n" + "="*70)
print("COMPARISON: CURRENT vs PROPOSED")
print("="*70)
print("""
Behavior                    | Current | Proposed | Target  | Status
────────────────────────────┼─────────┼──────────┼─────────┼────────
Perfect play (BRINE)        |  +2.62  |  +1.62   | +2 to 3 | Needs ⬆
Good opener (CRANE)         |  +1.99  |  +0.99   | +1 to 2 | ✅ OK
4-letter word               |  +0.78  |  -1.40   | < -0.5  | ✅ FIXED
Invalid word (XYZZZ)        |  +0.72  |  -1.40   | < -0.5  | ✅ FIXED
Reusing dead letters        |  +0.38  |  -0.62   | < -0.5  | ✅ FIXED
Ignoring feedback           |  +1.61  |  +0.61   | < -0.5  | Still bad
Hallucinating               |  -0.07  |  -1.07   | < -1.0  | ✅ FIXED
Lowercase                   |  +1.80  |  +0.80   | < -0.5  | Still bad

ISSUES REMAINING:
1. Good play rewards too low (need to boost feedback/info rewards)
2. "Ignoring feedback" still positive (need stronger feedback penalty)
3. Lowercase still positive (needs separate penalty)
""")

