from reward_functions import output_format_check, uses_previous_feedback, guess_value

def test_case(name, completion, history, secret):
    """Test a single case and show rewards"""
    example = {
        "word_list": "five_letter_words.csv",
        "past_guess_history": history,
        "secret_word": secret
    }
    
    format_r = output_format_check("", completion, example, training_progress=0.5)
    feedback_r = uses_previous_feedback("", completion, example)
    info_r = guess_value("", completion, example)
    total = format_r + feedback_r + info_r
    
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Format: {format_r:>6.2f} | Feedback: {feedback_r:>6.2f} | Info: {info_r:>6.2f} | TOTAL: {total:>6.2f}")
    
    if total > 1.0:
        print("🎉 STRONG POSITIVE - Model will repeat")
    elif total > 0.2:
        print("👍 Positive - Model may learn")
    elif total > -0.2:
        print("😐 Weak signal - Won't learn much")
    elif total > -1.0:
        print("👎 Negative - Model will avoid")
    else:
        print("❌ STRONG NEGATIVE - Model will strongly avoid")
    
    return total

print("\n" + "="*70)
print("REWARD BALANCE ANALYSIS")
print("="*70)

# Good behaviors that SHOULD be strongly rewarded
print("\n\n📊 GOOD BEHAVIORS (should be >1.0)")
print("─"*70)

test_case(
    "Perfect guess using all feedback correctly",
    "</think>\n<guess>BRINE</guess>",
    "[['CRANE', 'C(x) R(✓) A(x) N(x) E(✓)'], ['PRIME', 'P(x) R(✓) I(✓) M(x) E(✓)']]",
    "BRINE"
)

test_case(
    "Good strategic guess (CRANE opener)",
    "</think>\n<guess>CRANE</guess>",
    "[]",
    "BRAVE"
)

test_case(
    "Using (-) letters in new positions",
    "</think>\n<guess>SHARK</guess>",
    "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
    "SHARK"
)

# Bad behaviors that SHOULD be strongly penalized
print("\n\n📊 BAD BEHAVIORS (should be <-0.5)")
print("─"*70)

test_case(
    "4-letter word (invalid length)",
    "</think>\n<guess>CARE</guess>",
    "[]",
    "SCARE"
)

test_case(
    "Invalid word (not in dictionary)",
    "</think>\n<guess>XYZZZ</guess>",
    "[]",
    "BRAVE"
)

test_case(
    "Reusing 3 dead letters",
    "</think>\n<guess>CRAZE</guess>",
    "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
    "BRASS"
)

test_case(
    "Ignoring all (-) letters completely",
    "</think>\n<guess>SLOTH</guess>",
    "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
    "SHARP"
)

test_case(
    "Hallucinating fake feedback",
    "</think>\nGuess 3: BOOKS -> Feedback: B(✓) O(✓) O(✓) K(✓) S(-)\n<guess>BOOKS</guess>",
    "[['CRANE', 'C(x) R(x) A(-) N(-) E(x)'], ['BANJO', 'B(x) A(✓) N(✓) J(x) O(x)']]",
    "MANOR"
)

test_case(
    "Lowercase guess (wrong format)",
    "</think>\n<guess>board</guess>",
    "[['CRANE', 'C(x) R(x) A(✓) N(✓) E(x)']]",
    "BLANK"
)

# Edge cases
print("\n\n📊 EDGE CASES (should have clear signals)")
print("─"*70)

test_case(
    "Missing </think> tag but valid guess",
    "Analyzing feedback.\n<guess>PRESS</guess>",
    "[['CRANE', 'C(x) R(✓) A(x) N(x) E(x)']]",
    "PRESS"
)

test_case(
    "Extra text in <guess> tag",
    "</think>\n<guess> guessed-word: SLATE </guess>",
    "[]",
    "SLATE"
)

test_case(
    "Keeping ✓ positions but adding new letters",
    "</think>\n<guess>PRINT</guess>",
    "[['CRANE', 'C(x) R(✓) A(x) N(x) E(x)']]",
    "FROST"
)

# Summary
print("\n\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)
print("""
IDEAL REWARD RANGES:
  Perfect strategic play:     +2.0 to +3.0
  Good valid guess:           +1.0 to +2.0
  Acceptable but not great:   +0.2 to +1.0
  Weak/neutral:               -0.2 to +0.2
  Should avoid:               -1.0 to -0.2
  Must never do:              < -1.0

PROBLEMS TO FIX:
  1. Invalid length/words getting positive rewards
  2. Ignoring feedback getting positive rewards
  3. Hallucinations barely penalized
  4. Format reward (1.5) masks all problems
""")

