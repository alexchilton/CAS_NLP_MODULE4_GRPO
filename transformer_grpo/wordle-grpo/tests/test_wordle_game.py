"""
Unit tests for Wordle game utilities.

Tests the core game logic including guess validation, feedback formatting,
and helper functions.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from data.wordle_game import (
    validate_guess,
    format_feedback,
    is_valid_word,
    parse_feedback,
    extract_constraints_from_history,
    filter_candidates,
    is_winning_guess,
    get_letter_info,
    count_feedback_type,
    format_guess_history,
)


class TestValidateGuess:
    """Tests for validate_guess function."""

    def test_all_correct(self):
        """Test when all letters are correct."""
        secret = "CRANE"
        guess = "CRANE"
        feedback = validate_guess(secret, guess)
        assert feedback == "C(✓) R(✓) A(✓) N(✓) E(✓)"

    def test_all_wrong(self):
        """Test when no letters are in the word."""
        secret = "CRANE"
        guess = "BLOTS"
        feedback = validate_guess(secret, guess)
        assert feedback == "B(x) L(x) O(x) T(x) S(x)"

    def test_mixed_feedback(self):
        """Test with mixed correct, misplaced, and wrong letters."""
        secret = "CRANE"  # C(0) R(1) A(2) N(3) E(4)
        guess = "TRAIN"   # T(0) R(1) A(2) I(3) N(4)
        feedback = validate_guess(secret, guess)
        # T(x) R(✓) A(✓) I(x) N(-) - N is in CRANE but at position 3, not 4
        assert "R(✓)" in feedback
        assert "A(✓)" in feedback
        assert "N(-)" in feedback  # N is misplaced, not correct
        assert "T(x)" in feedback
        assert "I(x)" in feedback

    def test_misplaced_letters(self):
        """Test letters in wrong positions."""
        secret = "CRANE"
        guess = "NACRE"
        feedback = validate_guess(secret, guess)
        # N(-) A(-) C(-) R(-) E(✓)
        assert "E(✓)" in feedback
        assert "N(-)" in feedback
        assert "A(-)" in feedback
        assert "C(-)" in feedback
        assert "R(-)" in feedback

    def test_duplicate_letters_secret(self):
        """Test handling of duplicate letters in secret word."""
        secret = "ALLOW"
        guess = "LOLLY"
        feedback = validate_guess(secret, guess)
        # Should handle multiple L's correctly
        assert feedback.count("L") == 3  # Three L's in guess

    def test_duplicate_letters_guess(self):
        """Test handling of duplicate letters in guess word."""
        secret = "SPEED"  # Has two 'E's
        guess = "EERIE"   # Has three 'E's
        feedback = validate_guess(secret, guess)
        # Expected: The first two 'E's in the guess should be marked as misplaced (-),
        # as they match the two 'E's in the secret. The third 'E' in the guess
        # has no corresponding 'E' left in the secret, so it should be marked wrong (x).
        # R and I are not in the secret, so they are wrong (x).
        assert feedback == "E(-) E(-) R(x) I(x) E(x)"

    def test_raw_feedback(self):
        """Test raw feedback output as list."""
        secret = "CRANE"
        guess = "TRAIN"
        feedback = validate_guess(secret, guess, raw_feedback=True)
        assert isinstance(feedback, list)
        assert len(feedback) == 5


class TestFormatFeedback:
    """Tests for format_feedback function."""

    def test_format_with_all_symbols(self):
        """Test formatting with all symbol types."""
        feedback = "C(✓) R(✓) A(x) N(-) E(x)"
        formatted = format_feedback(feedback)
        assert "C✓" in formatted
        assert "R✓" in formatted
        assert "A✗" in formatted
        assert "N⚠" in formatted
        assert "E✗" in formatted

    def test_empty_feedback(self):
        """Test formatting empty feedback."""
        feedback = ""
        formatted = format_feedback(feedback)
        assert formatted == ""


class TestIsValidWord:
    """Tests for is_valid_word function."""

    def test_valid_word(self):
        """Test with a valid word."""
        word_list = ["CRANE", "TRAIN", "BRAIN"]
        assert is_valid_word("CRANE", word_list) is True

    def test_invalid_length(self):
        """Test with word of wrong length."""
        word_list = ["CRANE", "TRAIN"]
        assert is_valid_word("CAR", word_list) is False
        assert is_valid_word("CRANES", word_list) is False

    def test_not_in_list(self):
        """Test with word not in list."""
        word_list = ["CRANE", "TRAIN"]
        assert is_valid_word("BRAIN", word_list) is False

    def test_non_alphabetic(self):
        """Test with non-alphabetic characters."""
        word_list = ["CRANE"]
        assert is_valid_word("CR4NE", word_list) is False
        assert is_valid_word("CR@NE", word_list) is False

    def test_case_insensitive(self):
        """Test case insensitivity."""
        word_list = ["CRANE"]
        assert is_valid_word("crane", word_list) is True
        assert is_valid_word("Crane", word_list) is True

    def test_empty_or_none(self):
        """Test with empty or None input."""
        word_list = ["CRANE"]
        assert is_valid_word("", word_list) is False
        assert is_valid_word(None, word_list) is False


class TestParseFeedback:
    """Tests for parse_feedback function."""

    def test_parse_complete_feedback(self):
        """Test parsing complete feedback string."""
        feedback = "C(✓) R(-) A(x) N(✓) E(x)"
        parsed = parse_feedback(feedback)
        assert len(parsed) == 5
        assert parsed[0] == ('C', '✓')
        assert parsed[1] == ('R', '-')
        assert parsed[2] == ('A', 'x')
        assert parsed[3] == ('N', '✓')
        assert parsed[4] == ('E', 'x')

    def test_parse_empty_feedback(self):
        """Test parsing empty feedback."""
        feedback = ""
        parsed = parse_feedback(feedback)
        assert len(parsed) == 0


class TestExtractConstraintsFromHistory:
    """Tests for extract_constraints_from_history function."""

    def test_single_guess(self):
        """Test with single guess in history."""
        history = [("CRANE", "C(✓) R(-) A(x) N(x) E(x)")]
        correct, valid, wrong = extract_constraints_from_history(history)

        assert 'C' in correct
        assert 0 in correct['C']
        assert 'R' in valid
        assert 1 in valid['R']
        assert 'A' in wrong
        assert 'N' in wrong
        assert 'E' in wrong

    def test_multiple_guesses(self):
        """Test with multiple guesses in history."""
        history = [
            ("CRANE", "C(x) R(✓) A(✓) N(x) E(x)"),
            ("TRAIN", "T(x) R(✓) A(✓) I(x) N(-)")
        ]
        correct, valid, wrong = extract_constraints_from_history(history)

        assert 'R' in correct
        assert 'A' in correct
        assert 'N' in valid
        assert 'C' in wrong
        assert 'E' in wrong
        assert 'T' in wrong
        assert 'I' in wrong

    def test_empty_history(self):
        """Test with empty history."""
        history = []
        correct, valid, wrong = extract_constraints_from_history(history)

        assert len(correct) == 0
        assert len(valid) == 0
        assert len(wrong) == 0


class TestFilterCandidates:
    """Tests for filter_candidates function."""

    def test_filter_with_one_guess(self):
        """Test filtering with one guess."""
        # Simple test: if we guess "ABCDE" and all are wrong,
        # only words without those letters should remain
        candidates = ["FGHIJ", "ABCDE", "AFGHI", "KLMNO"]
        history = [("ABCDE", "A(x) B(x) C(x) D(x) E(x)")]
        filtered = filter_candidates(candidates, history)

        # FGHIJ has none of the wrong letters - should be included
        assert "FGHIJ" in filtered
        # ABCDE has all the wrong letters - should be excluded
        assert "ABCDE" not in filtered
        # AFGHI has A (which is wrong) - should be excluded
        assert "AFGHI" not in filtered
        # KLMNO has none of the wrong letters - should be included
        assert "KLMNO" in filtered

    def test_filter_with_multiple_guesses(self):
        """Test filtering with multiple guesses."""
        candidates = ["CRANE", "BRAIN", "TRAIN", "GRAIN", "DRAIN"]
        history = [
            ("SLATE", "S(x) L(x) A(-) T(x) E(x)"),
        ]
        filtered = filter_candidates(candidates, history)

        # Should only include words with A but not in position 2
        for word in filtered:
            assert 'A' in word
            assert word[2] != 'A'

    def test_filter_eliminates_all(self):
        """Test when filtering eliminates all candidates."""
        candidates = ["CRANE", "BRAIN"]
        history = [("SLOPE", "S(✓) L(✓) O(✓) P(✓) E(✓)")]
        filtered = filter_candidates(candidates, history)

        assert len(filtered) == 0


class TestIsWinningGuess:
    """Tests for is_winning_guess function."""

    def test_winning_feedback(self):
        """Test with all correct feedback."""
        feedback = "C(✓) R(✓) A(✓) N(✓) E(✓)"
        assert is_winning_guess(feedback) is True

    def test_non_winning_feedback(self):
        """Test with non-winning feedback."""
        feedback = "C(✓) R(✓) A(x) N(✓) E(✓)"
        assert is_winning_guess(feedback) is False

    def test_all_wrong_feedback(self):
        """Test with all wrong feedback."""
        feedback = "C(x) R(x) A(x) N(x) E(x)"
        assert is_winning_guess(feedback) is False


class TestGetLetterInfo:
    """Tests for get_letter_info function."""

    def test_valid_position(self):
        """Test getting info for valid position."""
        feedback = "C(✓) R(-) A(x) N(✓) E(x)"
        info = get_letter_info(feedback, 0)
        assert info == ('C', '✓')

        info = get_letter_info(feedback, 1)
        assert info == ('R', '-')

    def test_invalid_position(self):
        """Test with invalid position."""
        feedback = "C(✓) R(-) A(x) N(✓) E(x)"
        info = get_letter_info(feedback, 10)
        assert info is None

        info = get_letter_info(feedback, -1)
        assert info is None


class TestCountFeedbackType:
    """Tests for count_feedback_type function."""

    def test_count_correct(self):
        """Test counting correct letters."""
        feedback = "C(✓) R(✓) A(x) N(✓) E(x)"
        count = count_feedback_type(feedback, '✓')
        assert count == 3

    def test_count_misplaced(self):
        """Test counting misplaced letters."""
        feedback = "C(-) R(-) A(x) N(✓) E(x)"
        count = count_feedback_type(feedback, '-')
        assert count == 2

    def test_count_wrong(self):
        """Test counting wrong letters."""
        feedback = "C(x) R(x) A(x) N(x) E(x)"
        count = count_feedback_type(feedback, 'x')
        assert count == 5


class TestFormatGuessHistory:
    """Tests for format_guess_history function."""

    def test_format_with_history(self):
        """Test formatting with guess history."""
        history = [
            ("CRANE", "C(✓) R(✓) A(x) N(x) E(x)"),
            ("CRISP", "C(✓) R(✓) I(x) S(x) P(x)")
        ]
        formatted = format_guess_history(history)

        assert "Guess 1:" in formatted
        assert "Guess 2:" in formatted
        assert "CRANE" in formatted
        assert "CRISP" in formatted

    def test_format_empty_history(self):
        """Test formatting with empty history."""
        history = []
        formatted = format_guess_history(history)
        assert formatted == "No previous guesses"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
