"""
Basic tests for the rs_bpe package.
"""

import pytest
import rs_bpe


def test_package_metadata():
    """Test basic package metadata."""
    assert hasattr(rs_bpe, "__version__")
    assert isinstance(rs_bpe.__version__, str)
    assert len(rs_bpe.__version__.split(".")) >= 2


def test_module_structure():
    """Test that the module structure is as expected."""
    # The openai submodule should have the expected functions
    assert hasattr(rs_bpe, "cl100k_base")
    assert hasattr(rs_bpe, "o200k_base")
    assert hasattr(rs_bpe, "deepseek_base")
    assert hasattr(rs_bpe, "Tokenizer")
    assert hasattr(rs_bpe, "ParallelOptions")


def test_tokenizer_basic():
    """Test basic tokenizer functionality."""
    # Get the tokenizer
    tokenizer = rs_bpe.cl100k_base()

    # Test encoding
    tokens = tokenizer.encode("Hello, world!")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) > 0

    # Test decoding
    text = tokenizer.decode(tokens)
    assert text == "Hello, world!"

    # Test count
    count = tokenizer.count("Hello, world!")
    assert count == len(tokens)

    # Test count_till_limit with a limit equal to token count
    # Should return the count since it fits exactly
    exact_limit = count
    exact_count = tokenizer.count_till_limit("Hello, world!", exact_limit)
    assert exact_count == count

    # Test with a higher limit
    # Should still return the count
    high_limit = count + 5
    high_count = tokenizer.count_till_limit("Hello, world!", high_limit)
    assert high_count == count

    # Test with a lower limit
    # Should return None since it exceeds the limit
    low_limit = count - 1
    low_count = tokenizer.count_till_limit("Hello, world!", low_limit)
    assert low_count is None


@pytest.mark.parametrize(
    "input_string",
    [
        "Hello, world!",
        "Testing rs_bpe tokenizer.",
        "Another test string.",
        "Short",
        "A bit longer string for testing.",
        "😊 Unicode test!",
    ],
)
@pytest.mark.parametrize(
    "tokenizer_func",
    [
        rs_bpe.cl100k_base,
        rs_bpe.o200k_base,
        rs_bpe.deepseek_base,
    ],
)
def test_tokenizer_various_inputs(input_string, tokenizer_func):
    tokenizer = tokenizer_func()
    tokens = tokenizer.encode(input_string)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) > 0
    decoded_text = tokenizer.decode(tokens)
    assert decoded_text == input_string, f"tokens: {tokens}"
