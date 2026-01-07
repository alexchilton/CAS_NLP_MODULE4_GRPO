"""
Unit tests for reward functions.

Tests the three core reward functions and the CombinedReward class.
"""

import sys
from pathlib import Path
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
from training.reward_functions import (
    output_format_check,
    uses_previous_feedback,
    guess_value,
    CombinedReward,
)


# Fixtures

@pytest.fixture
def word_list():
    """Sample word list for testing."""
    return ["CRANE", "TRAIN", "BRAIN", "GRAIN", "DRAIN", "SLATE", "CRATE", "TRACE"]


@pytest.fixture
def word_list_file(word_list, tmp_path):
    """Create a temporary word list CSV file."""
    df = pd.DataFrame({"Word": word_list})
    file_path = tmp_path / "word_list.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def example_no_history(word_list):
    """Example with no past guesses."""
    return {
        "word_list": word_list,
        "past_guess_history": []
    }


@pytest.fixture
def example_with_history(word_list):
    """Example with past guess history."""
    return {
        "word_list": word_list,
        "past_guess_history": [
            ("SLATE", "S(x) L(x) A(-) T(x) E(x)"),
            ("CRANE", "C(x) R(✓) A(✓) I(x) N(x)"),
        ]
    }


@pytest.fixture
def example_with_correct_letters(word_list):
    """Example with some correct letters identified."""
    return {
        "word_list": word_list,
        "past_guess_history": [
            ("CRANE", "C(✓) R(✓) A(x) N(x) E(x)"),
        ]
    }


# Tests for output_format_check

class TestOutputFormatCheck:
    """Tests for output_format_check function."""

    def test_valid_format_valid_word(self, example_no_history):
        """Test valid format with valid word gets full reward."""
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"

        reward = output_format_check(prompt, completion, example_no_history)
        assert reward == 1.0

    def test_valid_format_invalid_word(self, example_no_history):
        """Test valid format with word not in list gets partial reward."""
        prompt = "Make a guess"
        completion = "I'll try ZZZZZ</think>\n<guess>ZZZZZ</guess>"

        reward = output_format_check(prompt, completion, example_no_history)
        assert reward == 0.5

    def test_valid_format_wrong_length(self, example_no_history):
        """Test valid format with wrong word length gets minimal reward."""
        prompt = "Make a guess"
        completion = "I'll try CAR</think>\n<guess>CAR</guess>"

        reward = output_format_check(prompt, completion, example_no_history)
        assert reward == 0.1

    def test_invalid_format_with_valid_word_gets_zero_reward(self, example_no_history):
        """Test that even a valid word gets zero reward if format is wrong."""
        prompt = "Make a guess"
        completion = "I'll try CRANE"  # No tags, but "CRANE" is in the word_list

        reward = output_format_check(prompt, completion, example_no_history)
        assert reward == 0.0

    def test_invalid_format_with_invalid_word_gets_zero_reward(self, example_no_history):
        """Test invalid format with no valid word gets zero reward."""
        prompt = "Make a guess"
        completion = "I'll try ZZZZZ"  # No tags, and "ZZZZZ" is not in the word_list

        reward = output_format_check(prompt, completion, example_no_history)
        assert reward == 0.0

    def test_invalid_format_missing_think_close(self, example_no_history):
        """Test invalid format missing </think> tag."""
        prompt = "Make a guess"
        completion = "I'll try CRANE\n<guess>CRANE</guess>"

        reward = output_format_check(prompt, completion, example_no_history)
        assert reward == 0.0

    def test_invalid_format_missing_guess_tags(self, example_no_history):
        """Test invalid format missing guess tags."""
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>"

        reward = output_format_check(prompt, completion, example_no_history)
        assert reward == 0.0

    def test_empty_completion(self, example_no_history):
        """Test empty completion gets zero reward."""
        prompt = "Make a guess"
        completion = ""

        reward = output_format_check(prompt, completion, example_no_history)
        assert reward == 0.0

    def test_case_insensitive(self, example_no_history):
        """Test that word matching is case-insensitive."""
        prompt = "Make a guess"
        completion = "I'll try crane</think>\n<guess>crane</guess>"

        reward = output_format_check(prompt, completion, example_no_history)
        assert reward == 1.0

    def test_with_word_list_file(self, word_list_file):
        """Test using word_list_path parameter."""
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"
        example = {"past_guess_history": []}

        reward = output_format_check(prompt, completion, example, word_list_path=word_list_file)
        assert reward == 1.0


# Tests for uses_previous_feedback

class TestUsesPreviousFeedback:
    """Tests for uses_previous_feedback function."""

    def test_no_history_gets_base_reward(self, example_no_history):
        """Test that no history gives base reward."""
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"

        reward = uses_previous_feedback(prompt, completion, example_no_history)
        assert reward == 0.1

    def test_respects_correct_positions(self, example_with_correct_letters):
        """Test positive reward for reusing correct letters in correct positions."""
        prompt = "Make a guess"
        # C and R are correct in positions 0 and 1
        completion = "I'll try CRISP</think>\n<guess>CRISP</guess>"

        reward = uses_previous_feedback(prompt, completion, example_with_correct_letters)
        # Expected: +0.2 (C) + 0.2 (R) + 0.05 (I) + 0.05 (S) + 0.05 (P) = 0.55
        assert abs(reward - 0.55) < 0.001

    def test_penalizes_wrong_letters(self, example_with_history):
        """Test penalty for using letters known to be wrong."""
        prompt = "Make a guess"
        # From history, wrong letters are: S, L, T, E, C, I, N
        completion = "I'll try SLATE</think>\n<guess>SLATE</guess>"

        reward = uses_previous_feedback(prompt, completion, example_with_history)
        # Expected: -0.5(S) -0.5(L) +0.2(A is correct) -0.5(T) -0.5(E) = -1.8
        assert abs(reward - (-1.8)) < 0.001

    def test_rewards_exploration(self, example_with_history):
        """Test reward for exploring new letters."""
        prompt = "Make a guess"
        # R and A are known good, F, U, D are new
        completion = "I'll try FRAUD</think>\n<guess>FRAUD</guess>"

        reward = uses_previous_feedback(prompt, completion, example_with_history)
        # Expected: +0.05(F) +0.2(R) +0.2(A) +0.05(U) +0.05(D) = 0.55
        # R and A are correct and in the right place from history.
        assert abs(reward - 0.55) < 0.001

    def test_invalid_format_returns_zero(self, example_with_history):
        """Test invalid format returns zero."""
        prompt = "Make a guess"
        completion = "CRANE"

        reward = uses_previous_feedback(prompt, completion, example_with_history)
        assert reward == 0.0

    def test_wrong_length_returns_zero(self, example_with_history):
        """Test wrong word length returns zero."""
        prompt = "Make a guess"
        completion = "I'll try CAR</think>\n<guess>CAR</guess>"

        reward = uses_previous_feedback(prompt, completion, example_with_history)
        assert reward == 0.0

    def test_empty_completion(self, example_with_history):
        """Test empty completion returns zero."""
        prompt = "Make a guess"
        completion = ""

        reward = uses_previous_feedback(prompt, completion, example_with_history)
        assert reward == 0.0

    def test_with_misplaced_letters(self):
        """Test handling of misplaced letters."""
        example = {
            "word_list": ["CRANE", "TRAIN"],
            "past_guess_history": [
                ("CRANE", "C(-) R(-) A(-) N(-) E(-)"),
            ]
        }
        prompt = "Make a guess"
        # Use C, R, A, N, E but in different positions
        completion = "I'll try NACRE</think>\n<guess>NACRE</guess>"

        reward = uses_previous_feedback(prompt, completion, example)
        # All letters are known to be in the word.
        # N, A, C, R are in new positions: 4 * 0.1 = 0.4
        # E is in the same wrong position: -0.2
        # Total = 0.4 - 0.2 = 0.2
        assert abs(reward - 0.2) < 0.001


# Tests for guess_value

class TestGuessValue:
    """Tests for guess_value function."""

    def test_valid_guess_in_word_list(self, example_no_history):
        """Test valid guess returns non-zero reward."""
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"

        reward = guess_value(prompt, completion, example_no_history)
        assert reward >= 0.0
        assert reward <= 1.0

    def test_invalid_word_returns_zero(self, example_no_history):
        """Test invalid word returns zero."""
        prompt = "Make a guess"
        completion = "I'll try ZZZZZ</think>\n<guess>ZZZZZ</guess>"

        reward = guess_value(prompt, completion, example_no_history)
        assert reward == 0.0

    def test_wrong_length_returns_zero(self, example_no_history):
        """Test wrong length returns zero."""
        prompt = "Make a guess"
        completion = "I'll try CAR</think>\n<guess>CAR</guess>"

        reward = guess_value(prompt, completion, example_no_history)
        assert reward == 0.0

    def test_invalid_format_returns_zero(self, example_no_history):
        """Test invalid format returns zero."""
        prompt = "Make a guess"
        completion = "CRANE"

        reward = guess_value(prompt, completion, example_no_history)
        assert reward == 0.0

    def test_with_history_filters_candidates(self, example_with_history):
        """Test that past history affects information gain calculation."""
        prompt = "Make a guess"
        completion = "I'll try BRAIN</think>\n<guess>BRAIN</guess>"

        reward = guess_value(prompt, completion, example_with_history)
        # Should return valid reward based on filtered candidates
        assert reward >= 0.0
        assert reward <= 1.0

    def test_good_guess_high_value(self):
        """Test that a strategically good guess gets high reward."""
        example = {
            "word_list": ["CRANE", "TRAIN", "BRAIN", "GRAIN", "DRAIN"],
            "past_guess_history": []
        }
        prompt = "Make a guess"
        # CRANE is generally a good first guess
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"

        reward = guess_value(prompt, completion, example)
        # Should get reasonable information gain
        assert reward > 0.0

    def test_empty_completion(self, example_no_history):
        """Test empty completion returns zero."""
        prompt = "Make a guess"
        completion = ""

        reward = guess_value(prompt, completion, example_no_history)
        assert reward == 0.0

    def test_with_word_list_file(self, word_list_file):
        """Test using word_list_path parameter."""
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"
        example = {"past_guess_history": []}

        reward = guess_value(prompt, completion, example, word_list_path=word_list_file)
        assert reward >= 0.0
        assert reward <= 1.0


# Tests for CombinedReward

class TestCombinedReward:
    """Tests for CombinedReward class."""

    def test_initialization(self):
        """Test CombinedReward initialization."""
        combined = CombinedReward(
            format_weight=1.0,
            feedback_weight=0.5,
            value_weight=0.3
        )
        assert combined.format_weight == 1.0
        assert combined.feedback_weight == 0.5
        assert combined.value_weight == 0.3

    def test_call_returns_combined_score(self, example_no_history):
        """Test that calling combined reward returns weighted sum."""
        combined = CombinedReward(
            format_weight=1.0,
            feedback_weight=0.0,
            value_weight=0.0
        )
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"

        reward = combined(prompt, completion, example_no_history)
        # With only format_weight=1.0, should equal output_format_check result
        expected = output_format_check(prompt, completion, example_no_history)
        assert reward == expected

    def test_weighted_combination(self, example_no_history):
        """Test weighted combination of rewards."""
        combined = CombinedReward(
            format_weight=1.0,
            feedback_weight=0.5,
            value_weight=0.3
        )
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"

        reward = combined(prompt, completion, example_no_history)

        # Manually compute expected
        format_r = output_format_check(prompt, completion, example_no_history)
        feedback_r = uses_previous_feedback(prompt, completion, example_no_history)
        value_r = guess_value(prompt, completion, example_no_history)
        expected = 1.0 * format_r + 0.5 * feedback_r + 0.3 * value_r

        assert abs(reward - expected) < 0.001

    def test_get_individual_rewards(self, example_no_history):
        """Test getting individual reward scores."""
        combined = CombinedReward()
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"

        rewards = combined.get_individual_rewards(prompt, completion, example_no_history)

        assert "format" in rewards
        assert "feedback" in rewards
        assert "value" in rewards
        assert all(isinstance(v, float) for v in rewards.values())

    def test_invalid_output_affects_all_components(self, example_no_history):
        """Test that invalid output gets low scores across all components."""
        combined = CombinedReward()
        prompt = "Make a guess"
        completion = "Invalid output"

        rewards = combined.get_individual_rewards(prompt, completion, example_no_history)

        # All should be zero or very low
        assert rewards["format"] == 0.0
        assert rewards["feedback"] == 0.0
        assert rewards["value"] == 0.0

    def test_with_word_list_file(self, word_list_file):
        """Test CombinedReward with word_list_path."""
        combined = CombinedReward(word_list_path=word_list_file)
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"
        example = {"past_guess_history": []}

        reward = combined(prompt, completion, example)
        assert reward > 0.0


# Edge case tests

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_malformed_xml_tags(self, example_no_history):
        """Test handling of malformed XML tags."""
        prompt = "Make a guess"
        completions = [
            "<think>test<think>\n<guess>CRANE</guess>",  # Unclosed think
            "<think>test</think>\n<guess>CRANE<guess>",  # Unclosed guess
            "<think>test</think><guess>CRANE</guess>",   # Missing newline
            "think>test</think>\n<guess>CRANE</guess>",  # Missing opening <
        ]

        for completion in completions:
            reward = output_format_check(prompt, completion, example_no_history)
            # Most malformed tags should return 0 or low score
            assert reward >= 0.0

    def test_whitespace_handling(self, example_no_history):
        """Test handling of various whitespace."""
        prompt = "Make a guess"
        completion = "  I'll try CRANE  </think>\n<guess>  CRANE  </guess>"

        reward = output_format_check(prompt, completion, example_no_history)
        # Should handle whitespace and still recognize CRANE
        assert reward == 1.0

    def test_example_missing_keys(self):
        """Test handling when example dict is missing expected keys."""
        prompt = "Make a guess"
        completion = "I'll try CRANE</think>\n<guess>CRANE</guess>"
        example = {}  # Missing word_list and past_guess_history

        # Should not crash
        format_r = output_format_check(prompt, completion, example)
        feedback_r = uses_previous_feedback(prompt, completion, example)
        value_r = guess_value(prompt, completion, example)

        # Should handle gracefully
        assert isinstance(format_r, float)
        assert isinstance(feedback_r, float)
        assert isinstance(value_r, float)

    def test_special_characters_in_guess(self, example_no_history):
        """Test handling of special characters in guess."""
        prompt = "Make a guess"
        completion = "I'll try CR@NE</think>\n<guess>CR@NE</guess>"

        reward = output_format_check(prompt, completion, example_no_history)
        # Should handle gracefully - likely return low score since not in word list
        assert reward >= 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
