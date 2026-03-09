"""
Kimi K2 benchmark at realistic long-context input lengths.

Target token counts (from a real workload distribution):
  P5:  15K tokens
  P25: 40K tokens
  P50: 75K tokens
  P75: 115K tokens
  P95: 175K tokens
"""

from __future__ import annotations

import gc
import statistics
import sys
import time

try:
    import rs_bpe
except ImportError:
    sys.exit("rs_bpe not available. Build with: maturin develop --release")

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Generate text that hits a target token count
# ---------------------------------------------------------------------------

# Base corpus: repeat README to get a large pool, then trim to target
try:
    with open("README.md") as f:
        _BASE = f.read()
except FileNotFoundError:
    _BASE = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
    ) * 500


def make_text_for_tokens(tok, target_tokens: int) -> str:
    """Build a string that encodes to approximately `target_tokens` tokens."""
    # Rough chars/token ratio
    ratio = len(_BASE) / max(len(tok.encode(_BASE)), 1)
    # Overshoot then trim
    needed_chars = int(target_tokens * ratio * 1.05)
    repeats = (needed_chars // len(_BASE)) + 1
    big = _BASE * repeats

    # Binary-search trim to hit the target
    lo, hi = 0, len(big)
    while lo < hi:
        mid = (lo + hi) // 2
        n = len(tok.encode(big[:mid]))
        if n < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    text = big[:lo]
    actual = len(tok.encode(text))
    return text, actual


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def time_encode(func, text: str, runs: int = 5, warmups: int = 2) -> float:
    for _ in range(warmups):
        func(text)
    times = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        func(text)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def time_encode_parallel(tok, text: str, chunk_size: int, opts, runs: int = 5, warmups: int = 2) -> float:
    for _ in range(warmups):
        tok.encode_split_chunks_parallel(text, chunk_size, opts)
    times = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        tok.encode_split_chunks_parallel(text, chunk_size, opts)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rs_tok = rs_bpe.kimi_k2()
    parallel_opts = rs_bpe.ParallelOptions(
        min_batch_size=20, chunk_size=100, max_threads=0, use_thread_pool=True,
    )
    num_threads = rs_bpe.get_num_threads()

    hf_tok = None
    if HF_AVAILABLE:
        print("Loading HuggingFace tokenizer...")
        hf_tok = AutoTokenizer.from_pretrained(
            "moonshotai/Kimi-K2-Instruct", trust_remote_code=True
        )

    targets = {
        "P5":  15_000,
        "P25": 40_000,
        "P50": 75_000,
        "P75": 115_000,
        "P95": 175_000,
    }

    runs = 5
    warmups = 2
    # For parallel split, use a chunk size in chars (~4.5 chars/tok * 2048 tok chunks)
    split_chunk_size = 8192

    print(f"Threads available: {num_threads}")
    print(f"Runs per case: {runs}, warmups: {warmups}")
    print(f"Parallel split_chunk_size: {split_chunk_size} chars")
    print()

    # Pre-generate all texts
    print("Generating test texts to hit target token counts...")
    texts: dict[str, tuple[str, int]] = {}
    for label, target in targets.items():
        text, actual = make_text_for_tokens(rs_tok, target)
        texts[label] = (text, actual)
        byte_size = len(text.encode("utf-8"))
        print(f"  {label}: target={target:,} actual={actual:,} tokens, {byte_size:,} bytes ({byte_size/1024:.0f} KB)")
    print()

    # Header
    header_parts = [
        f"{'Percentile':>10}",
        f"{'Tokens':>10}",
        f"{'Size':>10}",
    ]
    if hf_tok:
        header_parts.append(f"{'HF (ms)':>12}")
    header_parts.extend([
        f"{'rs_bpe (ms)':>12}",
        f"{'parallel (ms)':>14}",
    ])
    if hf_tok:
        header_parts.append(f"{'Speedup(seq)':>14}")
        header_parts.append(f"{'Speedup(par)':>14}")
    header_parts.extend([
        f"{'Tok/s (seq)':>14}",
        f"{'Tok/s (par)':>14}",
    ])
    print("".join(header_parts))
    print("-" * len("".join(header_parts)))

    for label, target in targets.items():
        text, actual_tokens = texts[label]
        byte_size = len(text.encode("utf-8"))
        size_str = f"{byte_size/1024:.0f} KB"

        # rs_bpe sequential
        rs_time = time_encode(rs_tok.encode, text, runs=runs, warmups=warmups)

        # rs_bpe parallel (split_chunks_parallel)
        par_time = time_encode_parallel(
            rs_tok, text, split_chunk_size, parallel_opts,
            runs=runs, warmups=warmups,
        )

        # HF
        hf_time = None
        if hf_tok:
            hf_time = time_encode(hf_tok.encode, text, runs=runs, warmups=warmups)

        rs_ms = rs_time * 1000
        par_ms = par_time * 1000
        rs_tps = actual_tokens / rs_time if rs_time > 0 else 0
        par_tps = actual_tokens / par_time if par_time > 0 else 0

        row = [
            f"{label:>10}",
            f"{actual_tokens:>10,}",
            f"{size_str:>10}",
        ]
        if hf_tok and hf_time is not None:
            hf_ms = hf_time * 1000
            row.append(f"{hf_ms:>12.1f}")
        row.extend([
            f"{rs_ms:>12.1f}",
            f"{par_ms:>14.1f}",
        ])
        if hf_tok and hf_time is not None:
            speedup_seq = hf_time / rs_time if rs_time > 0 else 0
            speedup_par = hf_time / par_time if par_time > 0 else 0
            row.append(f"{speedup_seq:>13.2f}x")
            row.append(f"{speedup_par:>13.2f}x")
        row.extend([
            f"{rs_tps:>14,.0f}",
            f"{par_tps:>14,.0f}",
        ])
        print("".join(row))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
