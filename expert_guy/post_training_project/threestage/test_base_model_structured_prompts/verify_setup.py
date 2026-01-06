"""
Quick verification that the prompt system works correctly
"""

from prompt_system import build_messages, get_feedback, GuessFeedback, SYSTEM_PROMPT
from wordle_game import WordleGame, extract_guess_from_completion, extract_thinking

def test_feedback_generation():
    """Test that feedback generation works correctly"""
    print("Testing feedback generation...")

    # Test case 1: All greens
    fb = get_feedback("CRANE", "CRANE")
    assert fb.feedback == "G G G G G", f"Expected all greens, got {fb.feedback}"
    print("✓ All greens works")

    # Test case 2: All grays
    fb = get_feedback("XYZAB", "CDEFG")
    assert fb.feedback == "X X X X X", f"Expected all grays, got {fb.feedback}"
    print("✓ All grays works")

    # Test case 3: Mixed
    fb = get_feedback("CRANE", "STARE")
    # C=X, R=Y, A=G, N=X, E=G
    assert fb.feedback.split()[2] == "G", "A should be green"
    assert fb.feedback.split()[4] == "G", "E should be green"
    assert fb.feedback.split()[1] == "Y", "R should be yellow"
    print("✓ Mixed feedback works")

    print("All feedback tests passed!\n")


def test_prompt_building():
    """Test that prompt building works correctly"""
    print("Testing prompt building...")

    # Test 1: First turn (no history)
    messages = build_messages([])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "first turn" in messages[1]["content"].lower()
    print("✓ First turn prompt works")

    # Test 2: After one guess
    fb1 = GuessFeedback("CRANE", "X X Y X Y")
    messages = build_messages([fb1])
    user_content = messages[1]["content"]

    assert "Current Knowledge" in user_content
    assert "Green" in user_content
    assert "Yellow" in user_content
    assert "Gray" in user_content
    assert "CRANE" in user_content
    print("✓ Multi-turn prompt works")

    # Verify state extraction
    assert "_ _ _ _ _" in user_content  # No greens yet
    assert "'A'" in user_content or "'E'" in user_content  # Yellows
    print("✓ State extraction works")

    print("All prompt tests passed!\n")


def test_guess_extraction():
    """Test guess extraction from model output"""
    print("Testing guess extraction...")

    # Test valid guess
    completion = "<think>This is my reasoning</think><guess>SLATE</guess>"
    guess = extract_guess_from_completion(completion)
    assert guess == "SLATE", f"Expected SLATE, got {guess}"
    print("✓ Valid guess extraction works")

    # Test with extra whitespace
    completion = "<think>reasoning</think><guess>  CRANE  </guess>"
    guess = extract_guess_from_completion(completion)
    assert guess == "CRANE", f"Expected CRANE, got {guess}"
    print("✓ Whitespace handling works")

    # Test with no guess
    completion = "<think>just thinking, no guess yet</think>"
    guess = extract_guess_from_completion(completion)
    assert guess is None, "Should return None when no guess found"
    print("✓ Missing guess handling works")

    print("All extraction tests passed!\n")


def test_game_flow():
    """Test full game flow"""
    print("Testing game flow...")

    word_list = {"CRANE", "SLATE", "STARE", "SOARE", "AUDIO"}
    game = WordleGame("CRANE", word_list)

    # Make a valid guess
    feedback, status = game.make_guess("SLATE")
    assert status == "continue", "Should continue after non-winning guess"
    print("✓ Valid guess works")

    # Try invalid guess (not in list)
    feedback, status = game.make_guess("ZZZZZ")
    assert status == "Invalid: not in word list"
    print("✓ Invalid word rejection works")

    # Try repeat guess
    feedback, status = game.make_guess("SLATE")
    assert "already guessed" in status
    print("✓ Repeat guess rejection works")

    # Win the game
    feedback, status = game.make_guess("CRANE")
    assert status == "win"
    assert feedback.feedback == "G G G G G"
    print("✓ Win detection works")

    print("All game flow tests passed!\n")


def show_example_prompt():
    """Show what a real prompt looks like"""
    print("="*80)
    print("EXAMPLE: What the model sees after 2 guesses")
    print("="*80)

    # Simulate game history
    fb1 = GuessFeedback("CRANE", "X X Y X Y")  # A and E are yellow
    fb2 = GuessFeedback("SLATE", "X G Y X G")  # L is green at pos 1, A is yellow, E is green at pos 4

    messages = build_messages([fb1, fb2])

    print("\nSYSTEM PROMPT (first 500 chars):")
    print(messages[0]["content"][:500] + "...\n")

    print("USER PROMPT (full):")
    print(messages[1]["content"])
    print("\n" + "="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VERIFICATION SCRIPT - Testing Structured Prompt System")
    print("="*80 + "\n")

    try:
        test_feedback_generation()
        test_prompt_building()
        test_guess_extraction()
        test_game_flow()
        show_example_prompt()

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print("\nThe system is ready to use. You can now run:")
        print("  python test_base_qwen.py --games 5 --temperature 0.1")
        print("\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
