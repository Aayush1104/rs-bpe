"""
OpenAI tokenizer interface for Python applications.

Key Components:
    cl100k_base(): Function that returns the cl100k tokenizer used by GPT-3.5/4
    o200k_base(): Function that returns the o200k tokenizer
    Tokenizer: The tokenizer class providing encoding and decoding methods
    is_cached_cl100k(): Function to check if cl100k tokenizer is already cached
    is_cached_o200k(): Function to check if o200k tokenizer is already cached

Project Dependencies:
    This file uses: rs_bpe.bpe.openai: The Rust OpenAI tokenizer implementation
"""

__all__ = [
    "Tokenizer",
    "cl100k_base",
    "deepseek_base",
    "is_cached_cl100k",
    "is_cached_deepseek",
    "is_cached_o200k",
    "o200k_base",
]

try:

    class _OpenAIProxy:
        def __getattr__(self, name):
            import importlib

            openai_module = importlib.import_module("rs_bpe.bpe.openai")
            return getattr(openai_module, name)

    openai = _OpenAIProxy()

    from rs_bpe.bpe.openai import (
        Tokenizer,
        cl100k_base,
        deepseek_base,
        is_cached_cl100k,
        is_cached_deepseek,
        is_cached_o200k,
        o200k_base,
    )
except ImportError:
    import sys

    print("Error: Failed to import OpenAI tokenizer module", file=sys.stderr)
    print("Make sure to build the extension with: maturin develop", file=sys.stderr)
