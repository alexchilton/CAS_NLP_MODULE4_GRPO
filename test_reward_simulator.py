"""
Reward Function Simulator - Test reward functions without running the LLM

This script simulates various model outputs and shows what rewards they would receive.
Use this to quickly understand and debug reward function behavior.
"""

import sys
from reward_functions import output_format_check, uses_previous_feedback, guess_value, word_accuracy_reward

# Test scenarios representing different model behaviors
TEST_CASES = [
    {
        "name": "Perfect Valid Guess (First Turn)",
        "completion": "Let me start with a common word with frequent letters.\n</think>\n<guess>CRANE</guess>",
        "prompt": "Make your first 5-letter word guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[]",
            "secret_word": "BRAVE"
        }
    },
    {
        "name": "Perfect Valid Guess (Second Turn, Using Feedback)",
        "completion": "Based on the feedback, R and A are in the word but wrong positions. Let me try them elsewhere.\n</think>\n<guess>BRASS</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
            "secret_word": "BRASS"
        }
    },
    {
        "name": "Missing closing </think> tag",
        "completion": "Let me analyze the feedback.\nR is at position 2 (correct)\nI need to find other letters\n<guess>PRESS</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(✓) A(x) N(x) E(x)']]",
            "secret_word": "PRESS"
        }
    },
    {
        "name": "Invalid format with extra text in <guess>",
        "completion": "Looking at the feedback.\n</think>\n<guess> guessed-word: SLATE </guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[]",
            "secret_word": "SLATE"
        }
    },
    {
        "name": "4-letter word (invalid length)",
        "completion": "Based on feedback.\n</think>\n<guess>CARE</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[]",
            "secret_word": "SCARE"
        }
    },
    {
        "name": "Invalid word (not in word list)",
        "completion": "Let me try this.\n</think>\n<guess>XYZZZ</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[]",
            "secret_word": "BRAVE"
        }
    },
    {
        "name": "Reusing dead letters (C and E marked as x)",
        "completion": "I'll try different letters.\n</think>\n<guess>CRAZE</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
            "secret_word": "BRASS"
        }
    },
    {
        "name": "Keeping correct positions (R at 2, I at 3, E at 5)",
        "completion": "Keep the correct letters.\n</think>\n<guess>BRINE</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(✓) A(x) N(x) E(✓)'], ['PRIME', 'P(x) R(✓) I(✓) M(x) E(✓)']]",
            "secret_word": "BRINE"
        }
    },
    {
        "name": "Ignoring (-) letters (R and A marked as -)",
        "completion": "Let me try completely new letters.\n</think>\n<guess>SLOTH</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
            "secret_word": "SHARP"
        }
    },
    {
        "name": "Multiple <think> tags",
        "completion": "</think>\n<think>Wait, let me reconsider.\n</think>\n<guess>TRAIN</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[]",
            "secret_word": "TRAIN"
        }
    },
    {
        "name": "ROUND 6: One letter away (FROSN vs FROST)",
        "completion": "Based on feedback, trying FROSN.\n</think>\n<guess>FROSN</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(✓) A(x) N(x) E(x)'], ['FROST', 'F(-) R(✓) O(✓) S(✓) T(-))']]",
            "secret_word": "FROST"
        }
    },
    {
        "name": "ROUND 6: Perfect match (FROST)",
        "completion": "This must be it!\n</think>\n<guess>FROST</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(✓) A(x) N(x) E(x)'], ['TROFS', 'T(-) R(✓) O(✓) F(-) S(✓)']]",
            "secret_word": "FROST"
        }
    },
    {
        "name": "ROUND 6: Dead letter gets 0 info-gain (C and E are dead)",
        "completion": "Let me try this.\n</think>\n<guess>CRAZE</guess>",
        "prompt": "Make a new guess.",
        "example": {
            "word_list": "five_letter_words.csv",
            "past_guess_history": "[['CRANE', 'C(x) R(-) A(-) N(x) E(x)']]",
            "secret_word": "BRASS"
        }
    }
]

def simulate_reward(test_case, training_progress=0.5):
    """Simulate reward calculation for a test case"""
    print(f"\n{'='*80}")
    print(f"TEST: {test_case['name']}")
    print(f"{'='*80}")
    print(f"Completion preview: {test_case['completion'][:100]}...")
    print(f"Past guess history: {test_case['example']['past_guess_history']}")
    print(f"Secret word: {test_case['example']['secret_word']}")

    # Calculate rewards
    format_reward = output_format_check(
        test_case['prompt'],
        test_case['completion'],
        test_case['example'],
        training_progress=training_progress
    )

    feedback_reward = uses_previous_feedback(
        test_case['prompt'],
        test_case['completion'],
        test_case['example']
    )

    info_gain_reward = guess_value(
        test_case['prompt'],
        test_case['completion'],
        test_case['example']
    )

    accuracy_reward = word_accuracy_reward(
        test_case['prompt'],
        test_case['completion'],
        test_case['example']
    )

    total_reward = format_reward + feedback_reward + info_gain_reward + accuracy_reward

    # Display results
    print(f"\n--- REWARDS ---")
    print(f"Format Reward:       {format_reward:>8.2f}")
    print(f"Feedback Reward:     {feedback_reward:>8.2f}")
    print(f"Info Gain Reward:    {info_gain_reward:>8.2f}")
    print(f"Accuracy Reward:     {accuracy_reward:>8.2f}  [ROUND 6 NEW]")
    print(f"{'─'*40}")
    print(f"TOTAL REWARD:        {total_reward:>8.2f}")

    # Interpretation
    print(f"\n--- INTERPRETATION ---")
    if format_reward == 0:
        print("⚠️  Format check FAILED - Regex didn't match!")
    elif format_reward < 0:
        print(f"❌ Format penalty: {format_reward} (invalid length or word)")
    elif format_reward == 1.0:
        print("✅ Format check PASSED")

    if feedback_reward < 0:
        print(f"❌ Negative feedback reward: {feedback_reward} (likely reused dead letters)")
    elif feedback_reward > 1.0:
        print(f"✅ Strong positive feedback: {feedback_reward} (good use of previous info)")
    elif feedback_reward > 0:
        print(f"👍 Positive feedback: {feedback_reward}")

    if total_reward > 2.0:
        print("\n🎉 STRONG POSITIVE SIGNAL - Model will repeat this behavior")
    elif total_reward > 0.5:
        print("\n👍 Positive signal - Model may learn this")
    elif total_reward > -0.5:
        print("\n😐 Weak/neutral signal - Model won't learn much")
    else:
        print("\n❌ NEGATIVE SIGNAL - Model will avoid this behavior")

    return {
        "name": test_case["name"],
        "format_reward": format_reward,
        "feedback_reward": feedback_reward,
        "info_gain_reward": info_gain_reward,
        "accuracy_reward": accuracy_reward,
        "total_reward": total_reward
    }

def run_simulation():
    """Run simulation on all test cases"""
    print("="*80)
    print("REWARD FUNCTION SIMULATOR")
    print("="*80)
    print("\nThis simulates various model outputs to understand reward behavior.")
    print("Training progress is set to 50% (mid-training).")

    results = []
    for test_case in TEST_CASES:
        result = simulate_reward(test_case, training_progress=0.5)
        results.append(result)

    # Summary table
    print("\n\n")
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Test Case':<50} {'Format':<10} {'Feedback':<10} {'Info':<10} {'Accuracy':<10} {'Total':<10}")
    print("─"*100)

    for result in results:
        print(f"{result['name']:<50} {result['format_reward']:<10.2f} {result['feedback_reward']:<10.2f} "
              f"{result['info_gain_reward']:<10.2f} {result['accuracy_reward']:<10.2f} {result['total_reward']:<10.2f}")

    # Analysis
    print("\n\n")
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)

    format_failures = sum(1 for r in results if r['format_reward'] == 0)
    negative_totals = sum(1 for r in results if r['total_reward'] < 0)
    positive_totals = sum(1 for r in results if r['total_reward'] > 1.0)

    print(f"Format failures (reward = 0): {format_failures}/{len(results)} ({100*format_failures/len(results):.0f}%)")
    print(f"Negative total rewards:       {negative_totals}/{len(results)} ({100*negative_totals/len(results):.0f}%)")
    print(f"Strong positive rewards (>1): {positive_totals}/{len(results)} ({100*positive_totals/len(results):.0f}%)")

    if format_failures > len(results) // 2:
        print("\n⚠️  WARNING: More than 50% of test cases failed format check!")
        print("   This means the model gets NO format reward for common outputs.")
        print("   The regex is too strict and needs to be fixed.")

    if negative_totals > positive_totals:
        print("\n⚠️  WARNING: More negative rewards than positive rewards!")
        print("   This suggests the model is being punished more than rewarded.")
        print("   Consider rebalancing the reward magnitudes.")

    avg_reward = sum(r['total_reward'] for r in results) / len(results)
    print(f"\nAverage total reward: {avg_reward:.2f}")

    if abs(avg_reward) < 0.5:
        print("⚠️  Average reward is near zero - weak learning signal!")

def interactive_mode():
    """Allow user to input custom completions"""
    print("\n\n")
    print("="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Enter your own completions to test (or 'quit' to exit)")

    while True:
        print("\n" + "─"*80)
        completion = input("Enter completion (after <think> tag): ")
        if completion.lower() in ['quit', 'exit', 'q']:
            break

        history = input("Past guess history (e.g., [['CRANE', 'C(x) R(-) A(-) N(x) E(x)']] or [] for none): ")
        secret = input("Secret word: ").upper()

        test_case = {
            "name": "Custom Test",
            "completion": completion,
            "prompt": "Make a new guess.",
            "example": {
                "word_list": "five_letter_words.csv",
                "past_guess_history": history if history else "[]",
                "secret_word": secret
            }
        }

        simulate_reward(test_case, training_progress=0.5)

if __name__ == "__main__":
    run_simulation()

    # Ask if user wants interactive mode
    print("\n")
    response = input("Would you like to test custom completions? (y/n): ")
    if response.lower() in ['y', 'yes']:
        interactive_mode()

    print("\n✓ Simulation complete!")
