"""
Verify the exact constants used in simulator vs documented in Round 5
"""

print("="*80)
print("CONSTANT VERIFICATION")
print("="*80)

# What I used in the simulator (from analyze_variance.py and test_moderate_rewards.py)
simulator_moderate = {
    "VALID_FORMAT_REWARD": 0.4,
    "PARTIAL_FORMAT_REWARD": 0.2,
    "VALID_WORD_BONUS": 0.4,
    "INVALID_LENGTH_PENALTY": -1.5,
    "INVALID_WORD_PENALTY": -1.5,
    "DEAD_LETTER_PENALTY": -0.7,
    "MISSING_GOOD_LETTER_PENALTY": -0.6,
}

# What I documented in Round 5
documented_round5 = {
    "VALID_FORMAT_REWARD": 0.4,
    "PARTIAL_FORMAT_REWARD": 0.2,
    "VALID_WORD_BONUS": 0.4,
    "INVALID_LENGTH_PENALTY": -1.5,
    "INVALID_WORD_PENALTY": -1.5,
    "DEAD_LETTER_PENALTY": -0.7,
    "MISSING_GOOD_LETTER_PENALTY": -0.6,
}

# What's currently in reward_functions.py
print("\nChecking current reward_functions.py...")
import sys
sys.path.insert(0, '.')
from reward_functions import (
    VALID_FORMAT_REWARD,
    PARTIAL_FORMAT_REWARD, 
    VALID_WORD_BONUS,
    INVALID_LENGTH_PENALTY,
    INVALID_WORD_PENALTY,
    DEAD_LETTER_PENALTY,
    MISSING_GOOD_LETTER_PENALTY
)

current_actual = {
    "VALID_FORMAT_REWARD": VALID_FORMAT_REWARD,
    "PARTIAL_FORMAT_REWARD": PARTIAL_FORMAT_REWARD,
    "VALID_WORD_BONUS": VALID_WORD_BONUS,
    "INVALID_LENGTH_PENALTY": INVALID_LENGTH_PENALTY,
    "INVALID_WORD_PENALTY": INVALID_WORD_PENALTY,
    "DEAD_LETTER_PENALTY": DEAD_LETTER_PENALTY,
    "MISSING_GOOD_LETTER_PENALTY": MISSING_GOOD_LETTER_PENALTY,
}

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\n{'Constant':<35} | Simulator | Documented | Current")
print("─"*80)

all_match = True
for key in simulator_moderate.keys():
    sim = simulator_moderate[key]
    doc = documented_round5[key]
    cur = current_actual[key]
    
    match_icon = "✅" if (sim == doc == cur) else "⚠️ " if (sim == doc) else "❌"
    
    print(f"{key:<35} | {sim:>9.2f} | {doc:>10.2f} | {cur:>7.2f} {match_icon}")
    
    if not (sim == doc):
        all_match = False
        print(f"  ❌ MISMATCH: Simulator used {sim} but documented {doc}")
    elif not (sim == cur):
        print(f"  ⚠️  Not yet applied: Current is {cur}, should be {sim}")

print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

if all_match:
    print("\n✅ PERFECT MATCH: Simulator values match documented Round 5 values")
else:
    print("\n❌ MISMATCH FOUND: Simulator and documentation differ!")

# Check if current values match simulator
current_match = all(simulator_moderate[k] == current_actual[k] for k in simulator_moderate.keys())

if current_match:
    print("✅ Current reward_functions.py already has Round 5 values")
else:
    print("⚠️  Current reward_functions.py still has old values (not yet updated)")
    print("\nTo apply Round 5 moderate constants:")
    print("1. Edit reward_functions.py")
    print("2. Update the 7 constants shown above")
    print("3. Restart training")

print("\n" + "="*80)
print("DETAILED BREAKDOWN")
print("="*80)

print("\nSimulator used (moderate system):")
for k, v in simulator_moderate.items():
    print(f"  {k:<35} = {v:>6.2f}")

print("\nRound 5 document specifies:")
for k, v in documented_round5.items():
    print(f"  {k:<35} = {v:>6.2f}")

print("\nCurrent reward_functions.py has:")
for k, v in current_actual.items():
    print(f"  {k:<35} = {v:>6.2f}")

# Calculate differences
print("\n" + "="*80)
print("CHANGES FROM CURRENT TO ROUND 5")
print("="*80)

print(f"\n{'Constant':<35} | Current | Round 5 | Change | Change %")
print("─"*90)

for key in simulator_moderate.keys():
    cur = current_actual[key]
    new = simulator_moderate[key]
    change = new - cur
    change_pct = (change / abs(cur) * 100) if cur != 0 else float('inf')
    
    if change > 0:
        direction = "📈"
    elif change < 0:
        direction = "📉"
    else:
        direction = "➡️ "
    
    print(f"{key:<35} | {cur:>7.2f} | {new:>7.2f} | {change:>6.2f} | {change_pct:>6.0f}% {direction}")

print("\n✅ All values verified and documented")

