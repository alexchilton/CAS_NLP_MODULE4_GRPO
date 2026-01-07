"""
Unit tests for reward functions to validate all fixes before training.

Tests cover:
1. Dead-letter penalty (cumulative punishment)
2. Staged invalid-guess penalties (early vs late training)
3. Position mask interpretation (correct vs wrong position)
4. Format validation (5-letter words only)
5. Valid word checking
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward_functions import (
    output_format_check,
    uses_previous_feedback,
    guess_value,
    WRONG_LETTER_PENALTY,
    INVALID_LENGTH_PENALTY_EARLY,
    INVALID_LENGTH_PENALTY_LATE,
    INVALID_WORD_PENALTY_EARLY,
    INVALID_WORD_PENALTY_LATE,
)

def test_dead_letter_cumulative_penalty():
    """Test that reusing dead letters gets escalating penalty"""
    print("\n" + "="*80)
    print("TEST 1: Dead Letter Cumulative Penalty")
    print("="*80)

    # Setup: Past guess shows T, O, M are all dead (x), R is correct
    example = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': "[('STORM', 'S(x) T(x) O(x) R(✓) M(x)')]",  # S, T, O, M are dead, R correct at pos 3
        'secret_word': 'CRANE'
    }

    # Test 1: Reuse 1 dead letter (T)
    completion1 = "Avoid dead letters\n</think>\n<guess>TRIKE</guess>"  # Contains T (dead)
    reward1 = uses_previous_feedback("", completion1, example)
    print(f"Test 1a: Reuse 1 dead letter (T) - Reward: {reward1}")
    assert reward1 < 0, f"Expected negative reward, got {reward1}"

    # Test 2: Reuse 2 dead letters (T, M)
    completion2 = "Try TRAMP\n</think>\n<guess>TRAMP</guess>"  # Contains T, M (both dead)
    reward2 = uses_previous_feedback("", completion2, example)
    print(f"Test 1b: Reuse 2 dead letters (T, M) - Reward: {reward2}")
    assert reward2 < reward1, f"Expected worse penalty for 2 dead letters: {reward2} vs {reward1}"

    # Test 3: Reuse 3 dead letters (T, O, M)
    completion3 = "Try STOMP\n</think>\n<guess>STOMP</guess>"  # Contains T, O, M (all dead)
    reward3 = uses_previous_feedback("", completion3, example)
    print(f"Test 1c: Reuse 3 dead letters (T, O, M) - Reward: {reward3}")
    assert reward3 < reward2, f"Expected worse penalty for 3 dead letters: {reward3} vs {reward2}"

    print("✓ PASSED: Dead letter penalty escalates with more violations")


def test_staged_invalid_length_penalty():
    """Test that invalid length penalties increase during training"""
    print("\n" + "="*80)
    print("TEST 2: Staged Invalid Length Penalty")
    print("="*80)

    example = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': [],
        'secret_word': 'CRANE'
    }

    # Test early training (progress = 0.0)
    completion = "Let me try BIG\n</think>\n<guess>BRIG</guess>"  # 4 letters
    reward_early = output_format_check("", completion, example, training_progress=0.0)
    print(f"Test 2a: Early training (0%) - 4-letter guess penalty: {reward_early}")
    assert abs(reward_early - INVALID_LENGTH_PENALTY_EARLY) < 0.01, f"Expected {INVALID_LENGTH_PENALTY_EARLY}, got {reward_early}"

    # Test late training (progress = 1.0)
    reward_late = output_format_check("", completion, example, training_progress=1.0)
    print(f"Test 2b: Late training (100%) - 4-letter guess penalty: {reward_late}")
    assert abs(reward_late - INVALID_LENGTH_PENALTY_LATE) < 0.01, f"Expected {INVALID_LENGTH_PENALTY_LATE}, got {reward_late}"

    # Test mid training (progress = 0.5)
    reward_mid = output_format_check("", completion, example, training_progress=0.5)
    print(f"Test 2c: Mid training (50%) - 4-letter guess penalty: {reward_mid}")
    expected_mid = INVALID_LENGTH_PENALTY_EARLY + (INVALID_LENGTH_PENALTY_LATE - INVALID_LENGTH_PENALTY_EARLY) * 0.5
    assert abs(reward_mid - expected_mid) < 0.01, f"Expected {expected_mid}, got {reward_mid}"

    assert reward_late < reward_mid < reward_early, "Penalties should increase during training"
    print("✓ PASSED: Invalid length penalties increase during training")


def test_staged_invalid_word_penalty():
    """Test that invalid word penalties increase during training"""
    print("\n" + "="*80)
    print("TEST 3: Staged Invalid Word Penalty")
    print("="*80)

    example = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': [],
        'secret_word': 'CRANE'
    }

    # Test with a non-word
    completion = "Let me try XYZQW\n</think>\n<guess>XYZQW</guess>"  # Not a real word

    # Early training
    reward_early = output_format_check("", completion, example, training_progress=0.0)
    print(f"Test 3a: Early training (0%) - Non-word penalty: {reward_early}")

    # Late training
    reward_late = output_format_check("", completion, example, training_progress=1.0)
    print(f"Test 3b: Late training (100%) - Non-word penalty: {reward_late}")

    assert reward_late < reward_early, f"Late penalty {reward_late} should be worse than early {reward_early}"
    print("✓ PASSED: Invalid word penalties increase during training")


def test_position_mask_correct_vs_wrong():
    """Test that model distinguishes between correct position (✓) and wrong position (-)"""
    print("\n" + "="*80)
    print("TEST 4: Position Mask Interpretation")
    print("="*80)

    # Test Case 1: Letter in correct position should be rewarded
    example1 = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': "[('STORM', 'S(✓) T(x) O(x) R(-) M(x)')]",  # S is correct at pos 0, R is in word but wrong position
        'secret_word': 'STARE'
    }

    # Good guess: Keep S at position 0, try R at different position
    completion_good = "Keep S at position 0, try R elsewhere\n</think>\n<guess>STARE</guess>"
    reward_good = uses_previous_feedback("", completion_good, example1)
    print(f"Test 4a: Keep S(✓) at pos 0, move R(-) - Reward: {reward_good}")

    # Bad guess: Move S from position 0 (should be penalized)
    completion_bad = "Let me try\n</think>\n<guess>TRADE</guess>"  # S moved from position 0
    reward_bad = uses_previous_feedback("", completion_bad, example1)
    print(f"Test 4b: Move S from correct position - Reward: {reward_bad}")

    assert reward_good > reward_bad, f"Keeping correct position should be rewarded: {reward_good} vs {reward_bad}"
    print("✓ PASSED: Model correctly distinguishes ✓ (correct) vs - (wrong position)")


def test_valid_5_letter_word():
    """Test that only valid 5-letter words get positive rewards"""
    print("\n" + "="*80)
    print("TEST 5: Valid 5-Letter Word Check")
    print("="*80)

    example = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': [],
        'secret_word': 'CRANE'
    }

    # Test valid word
    completion_valid = "Let me try CRANE\n</think>\n<guess>CRANE</guess>"
    reward_valid = output_format_check("", completion_valid, example, training_progress=0.5)
    print(f"Test 5a: Valid 5-letter word (CRANE) - Reward: {reward_valid}")
    assert reward_valid == 1.0, f"Expected 1.0 for valid word, got {reward_valid}"

    # Test invalid length
    completion_short = "Let me try CRAN\n</think>\n<guess>CRAN</guess>"
    reward_short = output_format_check("", completion_short, example, training_progress=0.5)
    print(f"Test 5b: 4-letter word (CRAN) - Reward: {reward_short}")
    assert reward_short < 0, f"Expected negative reward for wrong length, got {reward_short}"

    # Test non-word
    completion_nonword = "Let me try XYZQW\n</think>\n<guess>XYZQW</guess>"
    reward_nonword = output_format_check("", completion_nonword, example, training_progress=0.5)
    print(f"Test 5c: Non-word (XYZQW) - Reward: {reward_nonword}")
    assert reward_nonword < 0, f"Expected negative reward for non-word, got {reward_nonword}"

    print("✓ PASSED: Only valid 5-letter words get positive rewards")


def test_wrong_position_penalty():
    """Test that reusing same wrong position gets penalized"""
    print("\n" + "="*80)
    print("TEST 6: Wrong Position Reuse Penalty")
    print("="*80)

    example = {
        'word_list': 'five_letter_words.csv',
        'past_guess_history': "[('STORM', 'S(✓) T(x) O(x) R(-) M(x)')]",  # R is in word but NOT at position 3
        'secret_word': 'STARE'
    }

    # Bad: Try R at position 3 again (same wrong position)
    completion_bad = "Try R again\n</think>\n<guess>STORK</guess>"  # R at position 3 again
    reward_bad = uses_previous_feedback("", completion_bad, example)
    print(f"Test 6a: Reuse R at same wrong position 3 - Reward: {reward_bad}")

    # Good: Try R at different position
    completion_good = "Try R at different position\n</think>\n<guess>STARE</guess>"  # R at position 4
    reward_good = uses_previous_feedback("", completion_good, example)
    print(f"Test 6b: Move R to different position - Reward: {reward_good}")

    assert reward_good > reward_bad, f"Moving R to new position should be better: {reward_good} vs {reward_bad}"
    print("✓ PASSED: Reusing same wrong position gets penalized")


def run_all_tests():
    """Run all sanity-check tests"""
    print("\n" + "="*80)
    print("RUNNING REWARD FUNCTION SANITY CHECKS")
    print("="*80)

    try:
        test_dead_letter_cumulative_penalty()
        test_staged_invalid_length_penalty()
        test_staged_invalid_word_penalty()
        test_position_mask_correct_vs_wrong()
        test_valid_5_letter_word()
        test_wrong_position_penalty()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nReward functions are ready for training!")
        return True
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
