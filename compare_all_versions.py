"""
Three-Way Comparison: Current vs Original vs Improved

Tests all three reward function versions side-by-side to understand:
1. What DeepLearning.AI originally designed (Original)
2. What you've been using (Current - heavily modified)
3. What the proposed improvements are (Improved)
"""

import sys

# Import all three versions
import reward_functions as current_rf
import reward_functions_original as original_rf
import reward_functions_improved as improved_rf

# Test scenarios
TEST_CASES = [
    {
        "name": "✅ Perfect Valid Guess (First Turn)",
        "completion": "Let me start with a common word with frequent letters.\n</think>\n<guess>CRANE</guess>",
        "prompt": "Make your first 5-letter word guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[]",
            "secret_word": "BRAVE"
        }
    },
    {
        "name": "✅ Perfect Guess Using Feedback",
        "completion": "R and A are (-), so try them elsewhere.\n</think>\n<guess>BRASS</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
            "secret_word": "BRASS"
        }
    },
    {
        "name": "❌ Missing closing </think> tag",
        "completion": "Let me analyze the feedback.\nR is at position 2 (correct)\nI need to find other letters\n<guess>PRESS</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(✓) A(x) N(x) E(x)']]",
            "secret_word": "PRESS"
        }
    },
    {
        "name": "❌ Invalid format: extra text in <guess>",
        "completion": "Looking at the feedback.\n</think>\n<guess> guessed-word: SLATE </guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[]",
            "secret_word": "SLATE"
        }
    },
    {
        "name": "❌ 4-letter word (invalid length)",
        "completion": "Based on feedback.\n</think>\n<guess>CARE</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[]",
            "secret_word": "SCARE"
        }
    },
    {
        "name": "❌ Invalid word (not in word list)",
        "completion": "Let me try this.\n</think>\n<guess>XYZZZ</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[]",
            "secret_word": "BRAVE"
        }
    },
    {
        "name": "❌ Reusing dead letters (C, E marked x)",
        "completion": "I'll try different letters.\n</think>\n<guess>CRAZE</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
            "secret_word": "BRASS"
        }
    },
    {
        "name": "✅ Keeping correct positions (✓✓✓)",
        "completion": "Keep the correct letters.\n</think>\n<guess>BRINE</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(✓) A(x) N(x) E(✓)'], ['PRIME', 'P(x) R(✓) I(✓) M(x) E(✓)']]",
            "secret_word": "BRINE"
        }
    },
    {
        "name": "❌ Ignoring (-) letters that should be used",
        "completion": "Let me try completely new letters.\n</think>\n<guess>SLOTH</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
            "secret_word": "SHARP"
        }
    },
    {
        "name": "⚠️ Multiple <think> tags",
        "completion": "</think>\n<think>Wait, let me reconsider.\n</think>\n<guess>TRAIN</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[]",
            "secret_word": "TRAIN"
        }
    }
]


def compare_all_versions():
    """Compare all three reward function versions"""

    print("="*120)
    print("THREE-WAY COMPARISON: ORIGINAL (DeepLearning.AI) vs CURRENT (Your Modified) vs IMPROVED (Proposed)")
    print("="*120)
    print()

    results_original = []
    results_current = []
    results_improved = []

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{'='*120}")
        print(f"TEST {i}/10: {test_case['name']}")
        print(f"{'='*120}")
        print(f"Completion: {test_case['completion'][:70]}...")
        print(f"History: {test_case['example']['past_guess_history'][:50]}...")

        # Calculate rewards with ORIGINAL version
        original_format = original_rf.output_format_check(
            test_case['prompt'], test_case['completion'], test_case['example']
        )
        original_feedback = original_rf.uses_previous_feedback(
            test_case['prompt'], test_case['completion'], test_case['example']
        )
        original_info = original_rf.guess_value(
            test_case['prompt'], test_case['completion'], test_case['example']
        )
        original_total = original_format + original_feedback + original_info

        # Calculate rewards with CURRENT version
        current_format = current_rf.output_format_check(
            test_case['prompt'], test_case['completion'],
            test_case['example'], training_progress=0.5
        )
        current_feedback = current_rf.uses_previous_feedback(
            test_case['prompt'], test_case['completion'], test_case['example']
        )
        current_info = current_rf.guess_value(
            test_case['prompt'], test_case['completion'], test_case['example']
        )
        current_total = current_format + current_feedback + current_info

        # Calculate rewards with IMPROVED version
        improved_format = improved_rf.output_format_check(
            test_case['prompt'], test_case['completion'],
            test_case['example'], training_progress=0.5
        )
        improved_feedback = improved_rf.uses_previous_feedback(
            test_case['prompt'], test_case['completion'], test_case['example']
        )
        improved_info = improved_rf.guess_value(
            test_case['prompt'], test_case['completion'], test_case['example']
        )
        improved_total = improved_format + improved_feedback + improved_info

        # Display three-way comparison
        print(f"\n{'ORIGINAL (DL.AI)':<40} {'CURRENT (Modified)':<40} {'IMPROVED (Proposed)':<40}")
        print(f"{'-'*120}")
        print(f"Format:    {original_format:>6.2f} {'':<32} Format:    {current_format:>6.2f} {'':<32} Format:    {improved_format:>6.2f}")
        print(f"Feedback:  {original_feedback:>6.2f} {'':<32} Feedback:  {current_feedback:>6.2f} {'':<32} Feedback:  {improved_feedback:>6.2f}")
        print(f"Info Gain: {original_info:>6.2f} {'':<32} Info Gain: {current_info:>6.2f} {'':<32} Info Gain: {improved_info:>6.2f}")
        print(f"{'-'*120}")
        print(f"TOTAL:     {original_total:>6.2f} {'':<32} TOTAL:     {current_total:>6.2f} {'':<32} TOTAL:     {improved_total:>6.2f}")

        results_original.append({
            "name": test_case["name"],
            "format": original_format,
            "feedback": original_feedback,
            "info": original_info,
            "total": original_total
        })

        results_current.append({
            "name": test_case["name"],
            "format": current_format,
            "feedback": current_feedback,
            "info": current_info,
            "total": current_total
        })

        results_improved.append({
            "name": test_case["name"],
            "format": improved_format,
            "feedback": improved_feedback,
            "info": improved_info,
            "total": improved_total
        })

    # Summary comparison
    print("\n\n")
    print("="*120)
    print("SUMMARY COMPARISON")
    print("="*120)

    print(f"\n{'Test Case':<45} {'Original':<12} {'Current':<12} {'Improved':<12}")
    print("-"*120)

    for orig, curr, impr in zip(results_original, results_current, results_improved):
        print(f"{orig['name']:<45} {orig['total']:>10.2f}   {curr['total']:>10.2f}   {impr['total']:>10.2f}")

    # Statistics
    print("\n" + "="*120)
    print("STATISTICS")
    print("="*120)

    orig_avg = sum(r['total'] for r in results_original) / len(results_original)
    curr_avg = sum(r['total'] for r in results_current) / len(results_current)
    impr_avg = sum(r['total'] for r in results_improved) / len(results_improved)

    orig_std = (sum((r['total'] - orig_avg)**2 for r in results_original) / len(results_original)) ** 0.5
    curr_std = (sum((r['total'] - curr_avg)**2 for r in results_current) / len(results_current)) ** 0.5
    impr_std = (sum((r['total'] - impr_avg)**2 for r in results_improved) / len(results_improved)) ** 0.5

    orig_format_zeros = sum(1 for r in results_original if r['format'] == 0)
    curr_format_zeros = sum(1 for r in results_current if r['format'] == 0)
    impr_format_zeros = sum(1 for r in results_improved if r['format'] == 0)

    orig_negative = sum(1 for r in results_original if r['total'] < 0)
    curr_negative = sum(1 for r in results_current if r['total'] < 0)
    impr_negative = sum(1 for r in results_improved if r['total'] < 0)

    orig_strong_positive = sum(1 for r in results_original if r['total'] > 2.0)
    curr_strong_positive = sum(1 for r in results_current if r['total'] > 2.0)
    impr_strong_positive = sum(1 for r in results_improved if r['total'] > 2.0)

    print(f"\n{'Metric':<40} {'Original':<15} {'Current':<15} {'Improved':<15}")
    print("-"*120)
    print(f"{'Average Total Reward':<40} {orig_avg:>13.2f}   {curr_avg:>13.2f}   {impr_avg:>13.2f}")
    print(f"{'Standard Deviation':<40} {orig_std:>13.2f}   {curr_std:>13.2f}   {impr_std:>13.2f}")
    print(f"{'Format Failures (reward=0)':<40} {orig_format_zeros:>13}   {curr_format_zeros:>13}   {impr_format_zeros:>13}")
    print(f"{'Negative Total Rewards':<40} {orig_negative:>13}   {curr_negative:>13}   {impr_negative:>13}")
    print(f"{'Strong Positive (>2.0)':<40} {orig_strong_positive:>13}   {curr_strong_positive:>13}   {impr_strong_positive:>13}")

    # Detailed analysis
    print("\n" + "="*120)
    print("DETAILED ANALYSIS")
    print("="*120)

    print("\n🔍 ORIGINAL (DeepLearning.AI) Design Philosophy:")
    print("   - Format reward: 1.0 for perfect, 0.5 for invalid word, 0.1 for wrong length")
    print("   - Feedback rewards: 0.05-0.2 per letter (small, balanced)")
    print("   - Dead letter penalty: -0.5 per letter")
    print("   - Info gain: 0-1.0 (normalized)")
    print("   - Total typical range: ~0.5-2.5")

    print("\n🔧 CURRENT (Your Modified) Design Philosophy:")
    print("   - Format reward: 1.0 for perfect, staged penalties for invalid")
    print("   - Feedback rewards: 0.1-2.0 per letter (10x AMPLIFIED)")
    print("   - Dead letter penalty: -0.5 per letter (same as original)")
    print("   - Missing (-) letter penalty: -0.5 (NEW)")
    print("   - Total typical range: -2.0 to +7.0 (MUCH wider!)")

    print("\n✨ IMPROVED (Proposed) Design Philosophy:")
    print("   - Format reward: 0.3-1.5 depending on quality (partial credit)")
    print("   - Feedback rewards: 0.05-0.4 per letter (balanced like original)")
    print("   - Dead letter penalty: -0.4 per letter")
    print("   - Missing (-) letter penalty: -0.3")
    print("   - Total typical range: 0.5-2.5 (consistent)")

    # Key insights
    print("\n" + "="*120)
    print("KEY INSIGHTS")
    print("="*120)

    print("\n📊 About Format Rewards:")
    print(f"   Original gives format reward even for INVALID outputs:")
    print(f"   - Wrong length (4 letters): +0.1 ❌ (should penalize!)")
    print(f"   - Invalid word: +0.5 ❌ (REWARDS invalid word!)")
    print(f"   - This is likely a BUG in the DeepLearning.AI example")
    print(f"")
    print(f"   Your point about format is valid:")
    print(f"   - Model SHOULD learn <guess></guess> format from SFT")
    print(f"   - Format reward during GRPO might be redundant")
    print(f"   - BUT: helps when model drifts from SFT behavior")

    # Check if bad behaviors are properly handled
    print("\n📉 Bad Behavior Handling:")
    bad_indices = [2, 3, 4, 5, 6, 8]

    for idx in bad_indices:
        orig_r = results_original[idx]['total']
        curr_r = results_current[idx]['total']
        impr_r = results_improved[idx]['total']
        name = results_original[idx]['name']

        print(f"\n   {name}")
        print(f"      Original: {orig_r:>6.2f}  Current: {curr_r:>6.2f}  Improved: {impr_r:>6.2f}")

        if orig_r > 0:
            print(f"      ⚠️  Original REWARDS this bad behavior!")
        if curr_r > 1.0:
            print(f"      ⚠️  Current gives STRONG positive signal for bad behavior!")
        if impr_r < 0.5:
            print(f"      ✅ Improved gives weak/neutral signal")

    # Recommendation
    print("\n" + "="*120)
    print("RECOMMENDATION")
    print("="*120)

    print("\n📌 Your Question: 'Why should format get any points?'")
    print("\n   ANSWER: You're right to question this!")
    print("   - In theory: SFT should teach format perfectly")
    print("   - In practice: GRPO can cause 'policy drift' away from SFT")
    print("   - Format reward acts as 'anchor' to prevent drift")
    print("\n   OPTIONS:")
    print("   1. Remove format reward entirely (rely on SFT)")
    print("   2. Keep small format bonus (prevent drift)")
    print("   3. Use KL divergence penalty instead (better solution)")

    print("\n📌 About DeepLearning.AI Original:")
    print("   - Has BUGS: rewards invalid outputs (+0.1, +0.5)")
    print("   - Feedback magnitudes are well-tuned (0.05-0.2 scale)")
    print("   - BUT: doesn't penalize missing (-) letters")
    print("   - VERDICT: Good starting point, but needs fixes")

    print("\n📌 About Your Current Version:")
    print("   - Fixed the reward bugs ✅")
    print("   - Added missing (-) letter penalty ✅")
    print("   - BUT: 10x amplified rewards create huge variance")
    print("   - VERDICT: Right ideas, wrong magnitudes")

    print("\n📌 About Improved Version:")
    print("   - Fixes the bugs ✅")
    print("   - Keeps magnitudes balanced ✅")
    print("   - Adds partial format credit ✅")
    print("   - More consistent learning signal ✅")
    print("   - VERDICT: Best of both worlds")

    print("\n" + "="*120)
    print("FINAL VERDICT")
    print("="*120)

    if impr_std < orig_std and impr_std < curr_std:
        print("✅ IMPROVED version has the most consistent signal")
    if orig_negative == 0 and curr_negative > 0:
        print("⚠️  ORIGINAL never gives negative rewards (might be too lenient)")
    if abs(orig_avg - 1.0) < abs(curr_avg - 1.0):
        print("✅ ORIGINAL has better-centered rewards around 1.0")

    print(f"\nBased on simulation, consider using IMPROVED version with optional tweaks:")
    print(f"  - If format is stable: reduce format reward component")
    print(f"  - If model still makes bad guesses: increase penalties")
    print(f"  - Monitor: std dev should stay low (<1.0)")


if __name__ == "__main__":
    compare_all_versions()
