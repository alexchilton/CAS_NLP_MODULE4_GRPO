"""
Find OPTIMAL balance - aggressive penalties
"""

def optimal_rewards(has_guess, proper_format, word_len, valid_word, is_upper, fb_score, info_score):
    """Optimal reward calculations"""
    # Format - reduced base rewards
    if not has_guess:
        fmt = 0.0
    elif word_len != 5:
        fmt = (0.3 if proper_format else 0.1) - 2.5  # stronger length penalty
    elif not valid_word:
        fmt = (0.3 if proper_format else 0.1) - 2.5  # stronger invalid penalty
    elif not is_upper:
        fmt = (0.3 if proper_format else 0.1) + 0.3 - 2.0  # stronger lowercase penalty
    elif ":" in str(proper_format) or not proper_format:  # extra text or bad format
        fmt = 0.1 + 0.3 - 0.5  # small penalty for junk
    else:
        fmt = 0.3 + 0.3  # reduced base (was 0.5 + 0.5 = 1.0)
    
    # Feedback - 3x multiplier for strong signals
    fb = fb_score * 3.0
    
    # Info - 1.5x boost for good exploration
    info = info_score * 1.5
    
    return fmt, fb, info, fmt + fb + info

def test_opt(name, has_guess, proper_fmt, wlen, valid, is_up, fb, info, has_extra_text=False):
    fmt, fb_scaled, info_scaled, total = optimal_rewards(has_guess, proper_fmt, wlen, valid, is_up, fb, info)
    
    if has_extra_text:
        total -= 0.5  # extra penalty for junk
    
    status = "🎉" if total > 1.5 else "👍" if total > 0.5 else "😐" if total > -0.5 else "👎" if total > -1.5 else "❌"
    
    print(f"{name:<40} | {total:>6.2f} {status}")
    return total

print("="*80)
print("OPTIMAL REWARD SYSTEM (AGGRESSIVE)")
print("="*80)
print("\nChanges:")
print("  - Base format reward: 0.5 → 0.3")
print("  - Valid word bonus: 0.5 → 0.3")
print("  - Invalid length: -2.0 → -2.5")
print("  - Invalid word: -2.0 → -2.5")
print("  - Lowercase: -1.5 → -2.0")
print("  - Feedback multiplier: 3x")
print("  - Info multiplier: 1.5x")
print(f"\n{'Scenario':<40} | Total   Status")
print("─"*80)

print("\n🔴 FORMATTING")
fmt_scores = []
fmt_scores.append(test_opt("Lowercase 'scowl'", True, True, 5, True, False, -0.05, 0.20))
fmt_scores.append(test_opt("Lowercase 'board'", True, True, 5, True, False, 0.02, 0.10))
fmt_scores.append(test_opt("MiXeD CaSe", True, True, 5, True, False, 0.05, 0.15))
fmt_scores.append(test_opt("Dashes (invalid)", True, True, 5, False, True, 0.00, 0.00))
fmt_scores.append(test_opt("Extra text", True, True, 5, True, True, 0.05, 0.20, True))
fmt_scores.append(test_opt("Missing </think>", True, False, 5, True, True, 0.08, 0.15))
fmt_scores.append(test_opt("Wrong </thinking>", True, False, 5, True, True, 0.08, 0.15))
fmt_scores.append(test_opt("No think tags", True, False, 5, True, True, 0.05, 0.15))

print("\n🔴 INVALID WORDS")
inv_scores = []
inv_scores.append(test_opt("4 letters", True, True, 4, False, True, 0.00, 0.00))
inv_scores.append(test_opt("6 letters", True, True, 6, False, True, 0.00, 0.00))
inv_scores.append(test_opt("Nonsense", True, True, 5, False, True, 0.05, 0.00))

print("\n🔴 BAD STRATEGY")
bad_scores = []
bad_scores.append(test_opt("Reuse 2+ dead letters", True, True, 5, True, True, -1.15, 0.02))
bad_scores.append(test_opt("Ignore all (-)", True, True, 5, True, True, -0.35, 0.23))
bad_scores.append(test_opt("Don't keep (✓)", True, True, 5, True, True, -0.40, 0.22))
bad_scores.append(test_opt("Repeat guess", True, True, 5, True, True, -0.05, 0.00))
bad_scores.append(test_opt("Hallucinate feedback", True, True, 5, True, True, -1.90, 0.33))

print("\n🟢 EXCELLENT PLAY")
good_scores = []
good_scores.append(test_opt("Perfect feedback", True, True, 5, True, True, 0.85, 0.27))
good_scores.append(test_opt("Strong strategy", True, True, 5, True, True, 0.60, 0.30))

print("\n🟡 GOOD PLAY")
ok_scores = []
ok_scores.append(test_opt("Good opener", True, True, 5, True, True, 0.10, 0.39))
ok_scores.append(test_opt("(-) new position", True, True, 5, True, True, 0.25, 0.34))
ok_scores.append(test_opt("Keep ✓ + explore", True, True, 5, True, True, 0.15, 0.45))

print("\n" + "="*80)
print("RESULTS")
print("="*80)

total_bad = len(fmt_scores) + len(inv_scores) + len(bad_scores)
bad_penalized = sum(1 for s in fmt_scores + inv_scores + bad_scores if s < -0.5)
excellent = sum(1 for s in good_scores if s > 1.5)
good = sum(1 for s in ok_scores if s > 0.5 and s <= 1.5)

print(f"\nBad behaviors penalized: {bad_penalized}/{total_bad} ({100*bad_penalized/total_bad:.0f}%)")
print(f"Excellent play rewarded: {excellent}/{len(good_scores)}")
print(f"Good play rewarded: {good}/{len(ok_scores)}")

if bad_penalized >= total_bad * 0.85:
    print("\n✅ EXCELLENT - 85%+ bad behaviors penalized!")
    print("🎯 This configuration is ready for implementation")
else:
    print(f"\n⚠️  Still needs work - only {100*bad_penalized/total_bad:.0f}% coverage")

# Show actual numbers
print("\nKey stats:")
print(f"  Format issues avg: {sum(fmt_scores)/len(fmt_scores):.2f}")
print(f"  Invalid words avg: {sum(inv_scores)/len(inv_scores):.2f}")
print(f"  Bad strategy avg: {sum(bad_scores)/len(bad_scores):.2f}")
print(f"  Excellent play avg: {sum(good_scores)/len(good_scores):.2f}")
print(f"  Good play avg: {sum(ok_scores)/len(ok_scores):.2f}")

print("\n" + "="*80)
print("RECOMMENDED CONSTANTS FOR reward_functions.py")
print("="*80)
print("""
VALID_FORMAT_REWARD = 0.3
PARTIAL_FORMAT_REWARD = 0.1
VALID_WORD_BONUS = 0.3
INVALID_LENGTH_PENALTY = -2.5
INVALID_WORD_PENALTY = -2.5
LOWERCASE_PENALTY = -2.0
JUNK_TEXT_PENALTY = -0.5

# Feedback (apply 3x multiplier in uses_previous_feedback)
CORRECT_POSITION_REWARD = 0.4 * 3 = 1.2
DEAD_LETTER_PENALTY = -0.4 * 3 = -1.2
MISSING_GOOD_LETTER = -0.3 * 3 = -0.9

# Info gain (apply 1.5x in guess_value)
Scale all info_gain results by 1.5x
""")

