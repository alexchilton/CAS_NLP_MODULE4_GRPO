"""
Test more edge cases and problematic patterns from training
"""
from reward_functions import output_format_check, uses_previous_feedback, guess_value

def quick_test(name, completion, history, secret):
    example = {
        "word_list": "five_letter_words.csv",
        "past_guess_history": history,
        "secret_word": secret
    }
    
    fmt = output_format_check("", completion, example, training_progress=0.5)
    fb = uses_previous_feedback("", completion, example)
    info = guess_value("", completion, example)
    total = fmt + fb + info
    
    status = "🎉" if total > 1.0 else "👍" if total > 0.2 else "😐" if total > -0.2 else "👎" if total > -1.0 else "❌"
    
    print(f"{name:<45} | {total:>6.2f} {status}")
    return total

print("="*80)
print("COMPREHENSIVE EDGE CASE TESTING")
print("="*80)
print(f"\n{'Scenario':<45} | Total   Status")
print("─"*80)

print("\n🔴 FORMATTING ISSUES (should all be negative)")
scores = []

scores.append(quick_test(
    "Lowercase with spaces 'scowl'",
    "analyze...</think>\n<guess> scowl </guess>",
    "[['CRANE', 'C(x) R(x) A(x) N(✓) E(✓)']]",
    "SNAKE"
))

scores.append(quick_test(
    "Lowercase no spaces 'board'",
    "analyze...</think>\n<guess>board</guess>",
    "[['CRANE', 'C(x) R(x) A(✓) N(✓) E(x)']]",
    "BLANK"
))

scores.append(quick_test(
    "MiXeD CaSe 'BrAvE'",
    "analyze...</think>\n<guess>BrAvE</guess>",
    "[]",
    "BRAVE"
))

scores.append(quick_test(
    "Dashes 'S-L-A-N-T'",
    "analyze...</think>\n<guess>S-L-A-N-T</guess>",
    "[['CRANE', 'C(x) R(x) A(-) N(-) E(x)']]",
    "SLANT"
))

scores.append(quick_test(
    "Extra text 'guessed-word: SLATE'",
    "analyze...</think>\n<guess> guessed-word: SLATE </guess>",
    "[]",
    "SLATE"
))

scores.append(quick_test(
    "Missing </think> tag",
    "analyze...\n<guess>PRESS</guess>",
    "[['CRANE', 'C(x) R(✓) A(x) N(x) E(x)']]",
    "PRESS"
))

scores.append(quick_test(
    "Wrong tag </thinking>",
    "analyze...</thinking>\n<guess>STORM</guess>",
    "[]",
    "STORM"
))

scores.append(quick_test(
    "No <think> or </think> at all",
    "<guess>TRAIN</guess>",
    "[]",
    "TRAIN"
))

print("\n🔴 INVALID WORDS (should all be < -1.0)")

scores.append(quick_test(
    "4 letters 'CARE'",
    "analyze...</think>\n<guess>CARE</guess>",
    "[]",
    "SCARE"
))

scores.append(quick_test(
    "4 letters 'REAR'",
    "analyze...</think>\n<guess>REAR</guess>",
    "[]",
    "SPEAR"
))

scores.append(quick_test(
    "6 letters 'BRAINS'",
    "analyze...</think>\n<guess>BRAINS</guess>",
    "[]",
    "BRAIN"
))

scores.append(quick_test(
    "Nonsense 'XYZZZ'",
    "analyze...</think>\n<guess>XYZZZ</guess>",
    "[]",
    "BRAVE"
))

scores.append(quick_test(
    "Nonsense 'QQQqq'",
    "analyze...</think>\n<guess>QQQQ</guess>",
    "[]",
    "STORM"
))

print("\n🔴 IGNORING FEEDBACK (should be < -0.5)")

scores.append(quick_test(
    "Reusing 2 dead letters C,E in CRAZE",
    "analyze...</think>\n<guess>CRAZE</guess>",
    "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
    "BRASS"
))

scores.append(quick_test(
    "Ignoring all (-) letters R,A",
    "analyze...</think>\n<guess>SLOTH</guess>",
    "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
    "SHARP"
))

scores.append(quick_test(
    "Not keeping (✓) position - R at 2",
    "analyze...</think>\n<guess>BRAIN</guess>",
    "[['CRANE', 'C(x) R(✓) A(x) N(x) E(x)']]",
    "FROST"
))

scores.append(quick_test(
    "Repeating same guess CRANE",
    "analyze...</think>\n<guess>CRANE</guess>",
    "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
    "BRASS"
))

print("\n🟢 GOOD BEHAVIOR (should be > 1.0)")

scores.append(quick_test(
    "Perfect adherence to feedback",
    "analyze...</think>\n<guess>BRINE</guess>",
    "[['CRANE', 'C(x) R(✓) A(x) N(x) E(✓)'], ['PRIME', 'P(x) R(✓) I(✓) M(x) E(✓)']]",
    "BRINE"
))

scores.append(quick_test(
    "Good opening move CRANE",
    "analyze...</think>\n<guess>CRANE</guess>",
    "[]",
    "BRAVE"
))

scores.append(quick_test(
    "Using (-) in new position",
    "analyze...</think>\n<guess>SHARK</guess>",
    "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
    "SHARK"
))

scores.append(quick_test(
    "Keeping ✓ and adding new letters",
    "analyze...</think>\n<guess>PRINT</guess>",
    "[['CRANE', 'C(x) R(✓) A(x) N(x) E(x)']]",
    "FROST"
))

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

formatting_issues = scores[0:8]
invalid_words = scores[8:13]
ignoring_feedback = scores[13:17]
good_behavior = scores[17:21]

print(f"\nFormatting issues: avg = {sum(formatting_issues)/len(formatting_issues):.2f}")
print(f"  Should be < -0.5, got {sum(1 for s in formatting_issues if s < -0.5)}/{len(formatting_issues)} negative")

print(f"\nInvalid words: avg = {sum(invalid_words)/len(invalid_words):.2f}")
print(f"  Should be < -1.0, got {sum(1 for s in invalid_words if s < -1.0)}/{len(invalid_words)} strongly negative")

print(f"\nIgnoring feedback: avg = {sum(ignoring_feedback)/len(ignoring_feedback):.2f}")
print(f"  Should be < -0.5, got {sum(1 for s in ignoring_feedback if s < -0.5)}/{len(ignoring_feedback)} negative")

print(f"\nGood behavior: avg = {sum(good_behavior)/len(good_behavior):.2f}")
print(f"  Should be > 1.0, got {sum(1 for s in good_behavior if s > 1.0)}/{len(good_behavior)} positive")

total_bad = len(formatting_issues) + len(invalid_words) + len(ignoring_feedback)
total_bad_negative = sum(1 for s in formatting_issues + invalid_words + ignoring_feedback if s < 0)

print(f"\n📊 Overall: {total_bad_negative}/{total_bad} bad behaviors get negative rewards")
print(f"   That's {100*total_bad_negative/total_bad:.0f}% - should be 100%!")

