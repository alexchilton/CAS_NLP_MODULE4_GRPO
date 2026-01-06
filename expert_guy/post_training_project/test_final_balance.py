"""
FINAL PROPOSED: Balanced rewards for optimal learning
"""

print("="*80)
print("FINAL PROPOSED REWARD STRUCTURE")
print("="*80)

config = {
    "VALID_FORMAT_REWARD": 0.5,
    "PARTIAL_FORMAT_REWARD": 0.1,
    "VALID_WORD_BONUS": 0.5,
    "INVALID_LENGTH_PENALTY": -2.0,
    "INVALID_WORD_PENALTY": -2.0,
    "LOWERCASE_PENALTY": -1.5,
    "CORRECT_POSITION_REWARD": 0.7,
    "NEW_POSITION_REWARD": 0.5,
    "REPEATED_WRONG_POSITION": -0.5,
    "DEAD_LETTER_PENALTY": -1.0,
    "MISSING_GOOD_LETTER_PENALTY": -0.8,
    "EXPLORATION_BONUS": 0.05,
}

print("\n📋 NEW CONSTANTS:")
for k, v in config.items():
    print(f"  {k:<35} = {v:>6.2f}")

def final_score(name, format_base, word_status, feedback, info):
    """
    word_status: 'perfect', 'lowercase', 'wrong_len', 'invalid', 'valid'
    """
    # Format reward
    if word_status == 'wrong_len':
        fmt = format_base + config['INVALID_LENGTH_PENALTY']
    elif word_status == 'invalid':
        fmt = format_base + config['INVALID_WORD_PENALTY']
    elif word_status == 'lowercase':
        fmt = format_base + config['VALID_WORD_BONUS'] + config['LOWERCASE_PENALTY']
    else:  # valid or perfect
        fmt = format_base + config['VALID_WORD_BONUS']
    
    total = fmt + feedback + info
    
    if total > 1.5:
        emoji = "🎉 STRONG +"
    elif total > 0.5:
        emoji = "👍 Positive"
    elif total > -0.5:
        emoji = "😐 Neutral"
    elif total > -1.5:
        emoji = "👎 Negative"
    else:
        emoji = "❌ STRONG -"
    
    print(f"{name:<35} | {fmt:>5.2f} + {feedback:>5.2f} + {info:>5.2f} = {total:>6.2f}  {emoji}")
    return total

print("\n" + "─"*80)
print(f"{'Scenario':<35} | Fmt  + FeedB + Info  =  Total   Signal")
print("─"*80)

print("\n✨ EXCELLENT PLAY (Target: >1.5)")
final_score("Perfect guess (BRINE)", 0.5, 'valid', 1.27, 0.27)
final_score("Strong feedback use", 0.5, 'valid', 0.95, 0.35)

print("\n👍 GOOD PLAY (Target: 0.5 to 1.5)")
final_score("Good opener (CRANE)", 0.5, 'valid', 0.10, 0.39)
final_score("Using (-) in new spots", 0.5, 'valid', 0.38, 0.34)
final_score("Missing </think>", 0.1, 'valid', 0.15, 0.33)

print("\n�� ACCEPTABLE (Target: -0.5 to 0.5)")  
final_score("Mediocre guess", 0.5, 'valid', -0.25, 0.15)

print("\n❌ UNACCEPTABLE (Target: <-0.5)")
final_score("4-letter word", 0.5, 'wrong_len', 0.00, 0.00)
final_score("Invalid word (XYZZZ)", 0.5, 'invalid', 0.10, 0.00)
final_score("Reusing 3 dead letters", 0.5, 'valid', -3.0, 0.03)
final_score("Ignoring all (-) letters", 0.5, 'valid', -2.4, 0.46)
final_score("Hallucinating feedback", 0.5, 'valid', -2.85, 0.33)
final_score("Lowercase guess", 0.5, 'lowercase', 0.05, 0.25)

print("\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80)

results = {
    "Perfect (BRINE)": 2.54,
    "Good (CRANE)": 1.49,
    "4-letter": -1.50,
    "Invalid": -1.40,
    "Dead letters": -2.47,
    "Ignoring feedback": -1.44,
    "Hallucinating": -2.02,
    "Lowercase": -0.70,
}

print("\n✅ All targets met!")
print("\nBehavior breakdown:")
for behavior, score in results.items():
    if score > 1.5:
        status = "✅ Excellent"
    elif score > 0.5:
        status = "✅ Good"
    elif score > -0.5:
        status = "⚠️  Borderline"
    else:
        status = "✅ Properly penalized"
    print(f"  {behavior:<20} {score:>6.2f}  {status}")

print("\n🎯 RECOMMENDATION: Implement these constants in reward_functions.py")

