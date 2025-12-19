"""
Pre-training validation script that tests reward functions on 3 sample games.
Run this BEFORE training to ensure all fixes are working correctly.

This validates:
1. Prompt contamination fix (no STORM/BRAVE references)
2. Dead-letter penalties
3. Position mask interpretation
4. Staged penalties
5. Valid word checking
"""

import sys
from logger_setup import logger
from reward_functions import (
    output_format_check,
    uses_previous_feedback,
    guess_value,
)

def test_game_1_dead_letters():
    """Test Game 1: Ensure dead letters are properly penalized"""
    print("\n" + "="*80)
    print("VALIDATION GAME 1: Dead Letter Punishment")
    print("="*80)
    print("Secret: CRANE")
    print("Past guess: STORM → S(x) T(x) O(x) R(✓) M(x)")
    print("Expected: Model should NOT reuse S, T, O, M (all dead letters)")
    print()

    example = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': [
            ('STORM', 'S(x) T(x) O(x) R(✓) M(x)')
        ],
        'secret_word': 'CRANE'
    }

    # Good guess: Avoid all dead letters, keep R at position 3
    good_completion = "R is at position 3. Avoid S,T,O,M\n</think>\n<guess>CRANE</guess>"
    good_reward = uses_previous_feedback("", good_completion, example)
    print(f"✓ GOOD guess (CRANE - avoids dead letters): Reward = {good_reward}")

    # Bad guess: Reuses 2 dead letters (S, T)
    bad_completion = "Let me try\n</think>\n<guess>STARE</guess>"
    bad_reward = uses_previous_feedback("", bad_completion, example)
    print(f"✗ BAD guess (STARE - reuses S, T): Reward = {bad_reward}")

    assert bad_reward < good_reward, f"Bad guess should have worse reward: {bad_reward} vs {good_reward}"
    print("✓ PASSED: Dead letters properly penalized\n")
    return True


def test_game_2_position_masks():
    """Test Game 2: Ensure position masks prevent misinterpretation"""
    print("\n" + "="*80)
    print("VALIDATION GAME 2: Position Mask Interpretation")
    print("="*80)
    print("Secret: BRISK")
    print("Past guess: BRAKE → B(✓) R(✓) A(x) K(-) E(x)")
    print("Expected: Keep B at pos 0, R at pos 1, try K at different position")
    print()

    example = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': [
            ('BRAKE', 'B(✓) R(✓) A(x) K(-) E(x)')
        ],
        'secret_word': 'BRISK'
    }

    # Good guess: Keeps B, R in correct positions, moves K
    good_completion = "Keep B and R, move K\n</think>\n<guess>BRISK</guess>"
    good_reward = uses_previous_feedback("", good_completion, example)
    print(f"✓ GOOD guess (BRISK - correct positions): Reward = {good_reward}")

    # Bad guess: Moves R from correct position 1
    bad_completion = "Let me try\n</think>\n<guess>BRICK</guess>"
    bad_reward = uses_previous_feedback("", bad_completion, example)
    print(f"✗ BAD guess (BRICK - moves R): Reward = {bad_reward}")

    assert good_reward > bad_reward, f"Good guess should have better reward: {good_reward} vs {bad_reward}"
    print("✓ PASSED: Position masks correctly interpreted\n")
    return True


def test_game_3_invalid_guesses():
    """Test Game 3: Ensure invalid guesses get staged penalties"""
    print("\n" + "="*80)
    print("VALIDATION GAME 3: Staged Invalid-Guess Penalties")
    print("="*80)
    print("Secret: PLUMB")
    print("Testing invalid length and invalid words at different training stages")
    print()

    example = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': [],
        'secret_word': 'PLUMB'
    }

    # Test invalid length (4 letters)
    invalid_length = "Let me try\n</think>\n<guess>PLUM</guess>"
    penalty_early = output_format_check("", invalid_length, example, training_progress=0.0)
    penalty_late = output_format_check("", invalid_length, example, training_progress=1.0)
    print(f"Invalid length (PLUM):")
    print(f"  Early training (0%): Penalty = {penalty_early}")
    print(f"  Late training (100%): Penalty = {penalty_late}")
    assert penalty_late < penalty_early, f"Late penalty should be worse: {penalty_late} vs {penalty_early}"
    print(f"  ✓ Penalty increases from {penalty_early} to {penalty_late}")

    # Test invalid word (not in dictionary)
    invalid_word = "Let me try\n</think>\n<guess>XYZQW</guess>"
    penalty_early2 = output_format_check("", invalid_word, example, training_progress=0.0)
    penalty_late2 = output_format_check("", invalid_word, example, training_progress=1.0)
    print(f"\nInvalid word (XYZQW):")
    print(f"  Early training (0%): Penalty = {penalty_early2}")
    print(f"  Late training (100%): Penalty = {penalty_late2}")
    assert penalty_late2 < penalty_early2, f"Late penalty should be worse: {penalty_late2} vs {penalty_early2}"
    print(f"  ✓ Penalty increases from {penalty_early2} to {penalty_late2}")

    # Test valid guess
    valid_guess = "Let me try\n</think>\n<guess>PLUMB</guess>"
    reward_valid = output_format_check("", valid_guess, example, training_progress=0.5)
    print(f"\nValid guess (PLUMB): Reward = {reward_valid}")
    assert reward_valid == 1.0, f"Valid guess should get reward 1.0, got {reward_valid}"
    print("  ✓ Valid guess gets reward 1.0")

    print("✓ PASSED: Staged penalties work correctly\n")
    return True


def run_validation():
    """Run all validation tests"""
    print("\n" + "="*80)
    print("PRE-TRAINING VALIDATION - Testing All Fixes")
    print("="*80)
    print()

    try:
        # First run unit tests
        print("Step 1: Running unit tests...")
        from test_reward_functions import run_all_tests
        if not run_all_tests():
            print("\n❌ Unit tests failed! Fix issues before training.")
            return False

        # Then run game simulations
        print("\n" + "="*80)
        print("Step 2: Running game simulations...")
        print("="*80)

        test_game_1_dead_letters()
        test_game_2_position_masks()
        test_game_3_invalid_guesses()

        print("\n" + "="*80)
        print("ALL VALIDATIONS PASSED ✓")
        print("="*80)
        print("\n🎉 Your fixes are working correctly!")
        print("\nYou can now safely run training with:")
        print("  python grpo_local_data_peft.py")
        print("\nTraining will save model to: output2/wordle-grpo-peft/")
        print()
        return True

    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        print("\nPlease fix the issues before training.")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease fix the issues before training.")
        return False


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
