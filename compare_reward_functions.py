"""
A/B Comparison: Current vs Improved Reward Functions

Runs the same test cases through both reward function versions
and shows side-by-side comparison.
"""

import sys

# Import both versions
import reward_functions as current_rf
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


def compare_reward_functions():
    """Compare both reward function versions on all test cases"""

    print("="*100)
    print("A/B COMPARISON: CURRENT vs IMPROVED REWARD FUNCTIONS")
    print("="*100)
    print()

    results_current = []
    results_improved = []

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{'='*100}")
        print(f"TEST {i}/10: {test_case['name']}")
        print(f"{'='*100}")
        print(f"Completion: {test_case['completion'][:80]}...")
        print(f"History: {test_case['example']['past_guess_history'][:60]}...")

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

        # Display comparison
        print(f"\n{'CURRENT VERSION':<50} {'IMPROVED VERSION':<50}")
        print(f"{'-'*100}")
        print(f"Format:    {current_format:>6.2f} {'':>42} Format:    {improved_format:>6.2f}")
        print(f"Feedback:  {current_feedback:>6.2f} {'':>42} Feedback:  {improved_feedback:>6.2f}")
        print(f"Info Gain: {current_info:>6.2f} {'':>42} Info Gain: {improved_info:>6.2f}")
        print(f"{'-'*100}")
        print(f"TOTAL:     {current_total:>6.2f} {'':>42} TOTAL:     {improved_total:>6.2f}")

        # Show difference
        diff = improved_total - current_total
        if abs(diff) > 0.1:
            symbol = "📈" if diff > 0 else "📉"
            print(f"\n{symbol} DIFFERENCE: {diff:+.2f} {'(BETTER)' if diff > 0 else '(WORSE)'}")

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
    print("="*100)
    print("SUMMARY COMPARISON")
    print("="*100)

    print(f"\n{'Test Case':<45} {'Current':<12} {'Improved':<12} {'Diff':<12}")
    print("-"*100)

    for curr, impr in zip(results_current, results_improved):
        diff = impr['total'] - curr['total']
        symbol = "+" if diff > 0 else ""
        print(f"{curr['name']:<45} {curr['total']:>10.2f}   {impr['total']:>10.2f}   {symbol}{diff:>9.2f}")

    # Statistics
    print("\n" + "="*100)
    print("STATISTICS")
    print("="*100)

    curr_avg = sum(r['total'] for r in results_current) / len(results_current)
    impr_avg = sum(r['total'] for r in results_improved) / len(results_improved)

    curr_std = (sum((r['total'] - curr_avg)**2 for r in results_current) / len(results_current)) ** 0.5
    impr_std = (sum((r['total'] - impr_avg)**2 for r in results_improved) / len(results_improved)) ** 0.5

    curr_format_zeros = sum(1 for r in results_current if r['format'] == 0)
    impr_format_zeros = sum(1 for r in results_improved if r['format'] == 0)

    curr_negative = sum(1 for r in results_current if r['total'] < 0)
    impr_negative = sum(1 for r in results_improved if r['total'] < 0)

    curr_strong_positive = sum(1 for r in results_current if r['total'] > 2.0)
    impr_strong_positive = sum(1 for r in results_improved if r['total'] > 2.0)

    print(f"\n{'Metric':<40} {'Current':<15} {'Improved':<15} {'Change'}")
    print("-"*100)
    print(f"{'Average Total Reward':<40} {curr_avg:>13.2f}   {impr_avg:>13.2f}   {impr_avg-curr_avg:>+8.2f}")
    print(f"{'Standard Deviation':<40} {curr_std:>13.2f}   {impr_std:>13.2f}   {impr_std-curr_std:>+8.2f}")
    print(f"{'Format Failures (reward=0)':<40} {curr_format_zeros:>13}   {impr_format_zeros:>13}   {impr_format_zeros-curr_format_zeros:>+8}")
    print(f"{'Negative Total Rewards':<40} {curr_negative:>13}   {impr_negative:>13}   {impr_negative-curr_negative:>+8}")
    print(f"{'Strong Positive (>2.0)':<40} {curr_strong_positive:>13}   {impr_strong_positive:>13}   {impr_strong_positive-curr_strong_positive:>+8}")

    # Key improvements
    print("\n" + "="*100)
    print("KEY IMPROVEMENTS")
    print("="*100)

    improvements = []

    if impr_format_zeros < curr_format_zeros:
        improvements.append(f"✅ Reduced format failures by {curr_format_zeros - impr_format_zeros} cases")

    if abs(impr_std) < abs(curr_std):
        improvements.append(f"✅ More consistent rewards (std dev: {curr_std:.2f} → {impr_std:.2f})")

    if impr_avg > curr_avg:
        improvements.append(f"✅ Higher average reward ({curr_avg:.2f} → {impr_avg:.2f})")
    elif impr_avg < curr_avg:
        improvements.append(f"⚠️  Lower average reward (may need adjustment)")

    if impr_negative < curr_negative:
        improvements.append(f"✅ Fewer negative rewards ({curr_negative} → {impr_negative})")

    # Check if bad behaviors are properly penalized
    bad_behavior_tests = [2, 3, 4, 5, 6, 8]  # indices of "bad" test cases
    curr_bad_avg = sum(results_current[i]['total'] for i in bad_behavior_tests) / len(bad_behavior_tests)
    impr_bad_avg = sum(results_improved[i]['total'] for i in bad_behavior_tests) / len(bad_behavior_tests)

    if impr_bad_avg < curr_bad_avg:
        improvements.append(f"✅ Bad behaviors penalized more ({curr_bad_avg:.2f} → {impr_bad_avg:.2f})")

    # Check if good behaviors are properly rewarded
    good_behavior_tests = [0, 1, 7]  # indices of "good" test cases
    curr_good_avg = sum(results_current[i]['total'] for i in good_behavior_tests) / len(good_behavior_tests)
    impr_good_avg = sum(results_improved[i]['total'] for i in good_behavior_tests) / len(good_behavior_tests)

    if impr_good_avg > curr_good_avg:
        improvements.append(f"✅ Good behaviors rewarded more ({curr_good_avg:.2f} → {impr_good_avg:.2f})")

    for improvement in improvements:
        print(improvement)

    if not improvements:
        print("⚠️  No clear improvements detected - may need further tuning")

    # Recommendation
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)

    if len([i for i in improvements if i.startswith("✅")]) >= 3:
        print("✅ IMPROVED VERSION shows significant benefits!")
        print("   Consider replacing reward_functions.py with reward_functions_improved.py")
    elif len([i for i in improvements if i.startswith("✅")]) >= 1:
        print("⚠️  IMPROVED VERSION shows some benefits but may need tuning")
    else:
        print("❌ IMPROVED VERSION needs more work")


if __name__ == "__main__":
    compare_reward_functions()
