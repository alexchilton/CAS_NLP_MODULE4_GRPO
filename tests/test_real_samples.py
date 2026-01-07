"""
Test real samples from training log with CURRENT vs PROPOSED rewards
"""
from reward_functions import output_format_check, uses_previous_feedback, guess_value

# Proposed V3 constants
PROPOSED = {
    "VALID_FORMAT_REWARD": 0.5,
    "VALID_WORD_BONUS": 0.5,
    "INVALID_LENGTH_PENALTY": -2.0,
    "INVALID_WORD_PENALTY": -2.0,
    "DEAD_LETTER_PENALTY": -1.0,
    "MISSING_GOOD_LETTER_PENALTY": -0.8,
}

def test_real_sample(name, completion, history, secret, expected_current):
    """Test a real sample"""
    example = {
        "word_list": "five_letter_words.csv",
        "past_guess_history": history,
        "secret_word": secret
    }
    
    # Current rewards
    fmt_curr = output_format_check("", completion, example, training_progress=0.5)
    fb_curr = uses_previous_feedback("", completion, example)
    info_curr = guess_value("", completion, example)
    total_curr = fmt_curr + fb_curr + info_curr
    
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"Completion: {completion[:100]}...")
    print(f"\nCURRENT REWARDS:")
    print(f"  Format: {fmt_curr:>6.2f} | Feedback: {fb_curr:>6.2f} | Info: {info_curr:>6.2f} | TOTAL: {total_curr:>6.2f}")
    print(f"  Expected from log: {expected_current:>6.2f} (diff: {abs(total_curr-expected_current):>6.2f})")
    
    # Assessment
    if total_curr > 1.0:
        print(f"  ⚠️  Model LEARNS this behavior (strong positive)")
    elif total_curr > 0.2:
        print(f"  ⚠️  Model may learn this (positive)")
    elif total_curr > -0.5:
        print(f"  😐 Weak signal")
    else:
        print(f"  ✅ Model avoids this (negative)")
    
    # Quick analysis
    if "guess> scowl" in completion.lower():
        print(f"  📝 ISSUE: Lowercase 'scowl' - should be penalized heavily")
    if "EIGHT" in completion and "</thinking>" in completion:
        print(f"  📝 ISSUE: Wrong tag '</thinking>' instead of '</think>'")
    if "KIRAN" in completion:
        print(f"  📝 ISSUE: 'KIRAN' not a common word, may not be in dictionary")
    if "AIRY" in completion:
        print(f"  📝 ISSUE: 4-letter word 'AIRY' - invalid length")
    
    return total_curr

print("="*80)
print("TESTING REAL SAMPLES FROM TRAINING LOG")
print("="*80)
print("\nExtracting actual model outputs and checking reward behavior...")

# Sample 1: Lowercase "scowl"
test_real_sample(
    "Sample 1: Lowercase guess 'scowl'",
    "First, let's analyze the given information from Guess 1 (CRANE). We received two correct letters with checkmarks: 'A' and 'E'. Now we need to find words that fit these criteria for our next attempt.</think>\n<guess> scowl </guess>",
    "[['CRANE', 'C(x) R(x) A(x) N(✓) E(✓)']]",
    "SNAKE",
    1.47
)

# Sample 2: Wrong tag </thinking>
test_real_sample(
    "Sample 2: Wrong closing tag </thinking>",
    "From Guess 1's feedback (C(x), R(x), A(x), N(x), E(-)), we know that none of these letters can appear anywhere else except for 'E' which appears somewhere correctly as indicated by '-' symbol...</thinking>\n<guess>EIGHT</guess>",
    "[['CRANE', 'C(x) R(x) A(x) N(x) E(-)'], ['SMELT', 'S(x) M(x) E(-) L(x) T(-)']]",
    "TEETH",
    1.14
)

# Sample 3: Reusing dead letter 'E'
test_real_sample(
    "Sample 3: Reusing dead letter E in SLEET",
    "First, let's analyze what we know from Guess 1 (CRANE). We got all letters as 'x', meaning none of them were part of the solution. Now for Guess 2 (SMELT), only the last letter 'L' matched with '-'. So now our knowledge about possible solutions has narrowed down significantly.</think>\n<guess>SLEET</guess>",
    "[['CRANE', 'C(x) R(x) A(x) N(x) E(-)'], ['SMELT', 'S(x) M(x) E(-) L(x) T(-)']]",
    "BUTYL",
    0.23
)

# Sample 4: Good use of feedback
test_real_sample(
    "Sample 4: Good feedback usage (AUGHT)",
    "From Guess 1's feedback (C(x), R(x), A(-), N(x), E(x)), we know that 'CRANE' doesn't contain any correct letters except for 'A' which isn't placed correctly. So far, only 'A' could potentially fit into our next guess slot at index 4. Also from Guess 2's feedback (B(✓), L(x), O(x), A(-), T(-)) it looks like there might not be an 'L' or an 'O'. We also see now that 'BLOAT' contains no other letters besides possibly 'T', so let's try adding another potential candidate.</think>\n<guess>AUGHT</guess>",
    "[['CRANE', 'C(x) R(x) A(-) N(x) E(x)'], ['BLOAT', 'B(✓) L(x) O(x) A(-) T(-)']]",
    "BATCH",
    2.12
)

# Sample 5: 4-letter word AIRY
test_real_sample(
    "Sample 5: Invalid 4-letter word 'AIRY'",
    "From Guess 1's feedback (C(x), R(x), A(✓), N(-), E(x)), we know that 'A' appears at index 4 as it gave us a correct match there. Also from Guess 2's feedback (I(-), N(-), A(✓), P(x), T(-)) ,we can see 'P' cannot appear anywhere because all other positions had X responses except for Index 0 where an I appeared which matched with our second attempt Inapt. So now only letters left could possibly fit into remaining indices.</think>\n<guess>AIRY</guess>",
    "[['CRANE', 'C(x) R(x) A(✓) N(-) E(x)'], ['INAPT', 'I(-) N(-) A(✓) P(x) T(-)']]",
    "SWANK",
    -0.52
)

# Sample 6: Possibly invalid word KIRAN
test_real_sample(
    "Sample 6: Uncommon/invalid word 'KIRAN'",
    "First let's analyze what we know from Guess 1 (CRANE). We can see that 'R' appears correctly at index 0. From Guess 2 (KINGS), we learn that none of these letters appear anywhere else except for possibly an 'I'. Also, since there were no checks on other positions like '-', it means they should not match with any previously seen correct or incorrect characters.</think>\n<guess>KIRAN</guess>",
    "[['CRANE', 'C(x) R(x) A(x) N(-) E(x)'], ['KINGS', 'K(x) I(✓) N(-) G(-) S(x)']]",
    "VITAL",
    -0.10
)

print("\n\n" + "="*80)
print("SUMMARY OF REAL TRAINING SAMPLES")
print("="*80)
print("""
PROBLEMS IDENTIFIED:
1. Lowercase 'scowl' gets +1.47 (should be negative)
2. Wrong tag '</thinking>' gets +1.14 (partial credit acceptable)
3. Reusing dead letters gets +0.23 (too lenient)
4. 4-letter word 'AIRY' gets -0.52 (good, but could be stronger)
5. Invalid 'KIRAN' gets -0.10 (too weak)

GOOD EXAMPLES:
1. Good feedback usage gets +2.12 (excellent!)

RECOMMENDED ACTIONS:
1. Implement proposed reward structure to fix these issues
2. Add specific lowercase penalty
3. Increase invalid word penalties
4. Scale up feedback penalties for ignoring hints
""")

