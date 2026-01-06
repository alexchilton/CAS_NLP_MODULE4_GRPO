"""
Simulate what would happen with PROPOSED reward system
We'll manually calculate since we can't modify the actual functions
"""

def proposed_rewards(has_guess, proper_format, word_len, valid_word, is_upper, fb_score, info_score):
    """Simulate proposed reward calculations"""
    # Format
    if not has_guess:
        fmt = 0.0
    elif word_len != 5:
        fmt = (0.5 if proper_format else 0.1) - 2.0
    elif not valid_word:
        fmt = (0.5 if proper_format else 0.1) - 2.0
    elif not is_upper:
        fmt = (0.5 if proper_format else 0.1) + 0.5 - 1.5  # +bonus -lowercase_penalty
    else:
        fmt = (0.5 if proper_format else 0.1) + 0.5
    
    # Feedback (scale by 2x for stronger signals)
    fb = fb_score * 2.0
    
    # Info (keep same)
    info = info_score
    
    return fmt, fb, info, fmt + fb + info

def test_proposed(name, has_guess, proper_fmt, wlen, valid, is_up, fb, info):
    fmt, fb_scaled, info_scaled, total = proposed_rewards(has_guess, proper_fmt, wlen, valid, is_up, fb, info)
    
    status = "🎉" if total > 1.0 else "👍" if total > 0.2 else "😐" if total > -0.2 else "👎" if total > -1.0 else "❌"
    
    print(f"{name:<40} | {total:>6.2f} {status}")
    return total

print("="*80)
print("PROPOSED REWARD SYSTEM SIMULATION")
print("="*80)
print("\nChanges:")
print("  - Format reward: 1.0 → 0.5")
print("  - Invalid length penalty: -0.3 → -2.0")
print("  - Invalid word penalty: -0.5 → -2.0")
print("  - Lowercase penalty: NEW -1.5")
print("  - Feedback penalties: 2x multiplier")
print(f"\n{'Scenario':<40} | Total   Status")
print("─"*80)

print("\n🔴 FORMATTING ISSUES")
scores = []

scores.append(test_proposed("Lowercase 'scowl'", True, True, 5, True, False, -0.05, 0.20))
scores.append(test_proposed("Lowercase 'board'", True, True, 5, True, False, 0.02, 0.10))
scores.append(test_proposed("MiXeD CaSe", True, True, 5, True, False, 0.05, 0.15))
scores.append(test_proposed("Dashes 'S-L-A-N-T'", True, True, 5, False, True, 0.00, 0.00))
scores.append(test_proposed("Extra text in guess", True, True, 5, True, True, 0.05, 0.20))
scores.append(test_proposed("Missing </think>", True, False, 5, True, True, 0.08, 0.15))
scores.append(test_proposed("Wrong </thinking>", True, False, 5, True, True, 0.08, 0.15))
scores.append(test_proposed("No think tags", True, False, 5, True, True, 0.05, 0.15))

print("\n🔴 INVALID WORDS")

scores.append(test_proposed("4 letters 'CARE'", True, True, 4, False, True, 0.00, 0.00))
scores.append(test_proposed("4 letters 'REAR'", True, True, 4, False, True, 0.00, 0.00))
scores.append(test_proposed("6 letters 'BRAINS'", True, True, 6, False, True, 0.00, 0.00))
scores.append(test_proposed("Nonsense 'XYZZZ'", True, True, 5, False, True, 0.05, 0.00))
scores.append(test_proposed("Nonsense 'QQQQQ'", True, True, 5, False, True, 0.05, 0.00))

print("\n🔴 IGNORING FEEDBACK")

scores.append(test_proposed("Reusing dead letters", True, True, 5, True, True, -1.15, 0.02))
scores.append(test_proposed("Ignoring (-) letters", True, True, 5, True, True, -0.35, 0.23))
scores.append(test_proposed("Not keeping (✓)", True, True, 5, True, True, -0.40, 0.22))
scores.append(test_proposed("Repeat same guess", True, True, 5, True, True, -0.05, 0.00))

print("\n🟢 GOOD BEHAVIOR")

scores.append(test_proposed("Perfect feedback use", True, True, 5, True, True, 0.85, 0.27))
scores.append(test_proposed("Good opener", True, True, 5, True, True, 0.10, 0.39))
scores.append(test_proposed("(-) in new position", True, True, 5, True, True, 0.25, 0.34))
scores.append(test_proposed("Keep ✓ + new letters", True, True, 5, True, True, 0.15, 0.45))

print("\n" + "="*80)
print("PROPOSED SYSTEM ANALYSIS")
print("="*80)

formatting = scores[0:8]
invalid = scores[8:13]
ignoring = scores[13:17]
good = scores[17:21]

fmt_neg = sum(1 for s in formatting if s < -0.5)
inv_neg = sum(1 for s in invalid if s < -1.0)
ign_neg = sum(1 for s in ignoring if s < -0.5)
good_pos = sum(1 for s in good if s > 1.0)

print(f"\nFormatting: {fmt_neg}/8 properly penalized")
print(f"Invalid words: {inv_neg}/5 strongly penalized")
print(f"Ignoring feedback: {ign_neg}/4 penalized")
print(f"Good behavior: {good_pos}/4 rewarded")

total_bad = 8 + 5 + 4
bad_negative = fmt_neg + inv_neg + ign_neg
print(f"\n📊 Overall: {bad_negative}/{total_bad} bad behaviors penalized ({100*bad_negative/total_bad:.0f}%)")

if bad_negative >= total_bad * 0.9:
    print("✅ EXCELLENT - 90%+ coverage!")
elif bad_negative >= total_bad * 0.75:
    print("👍 GOOD - 75%+ coverage")
else:
    print("⚠️  NEEDS WORK - <75% coverage")

