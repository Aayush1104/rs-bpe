"""
Python bindings for the BPE (Byte Pair Encoding) Rust implementation.

Key Components:
    bpe: Module providing the core BPE functionality
    openai: Module providing OpenAI tokenizers
    BytePairEncoding: The core BPE implementation class

Project Dependencies:
    This file uses: rs_bpe.bpe: The Rust extension module
"""

from .bpe import (
    ParallelOptions,
    Tokenizer,
    cl100k_base,
    deepseek_base,
    get_num_threads,
    is_cached_cl100k,
    is_cached_deepseek,
    is_cached_o200k,
    o200k_base,
)

# Package metadata
__version__ = "0.1.0"
__all__ = [
    "ParallelOptions",
    "Tokenizer",
    "cl100k_base",
    "deepseek_base",
    "get_num_threads",
    "is_cached_cl100k",
    "is_cached_deepseek",
    "is_cached_o200k",
    "o200k_base",
]
