"""
Kimi K2 Tokenizer Optimization Benchmark

Mirrors Cursor production traffic patterns:
  - Input: P50 = 75K tokens, P95 = 175K tokens
  - Output: P50 = 310 tokens (small)
  - Cache hit rate: ~95%
  - 75% of sessions are multi-turn (median ~25 min, P50 gap = 3 min)
  - 25% single-turn

Variants tested:
  1. Baseline:       apply_chat_template() → encode()
  2. +Direct:        tokenize_messages_direct()
  3. +Caching:       KimiChatEncoder.encode_messages()
  4. +Parallel:      (integrated into KimiChatEncoder for cold starts)
  5. +Buffer reuse:  (integrated into KimiChatEncoder across calls)
"""

from __future__ import annotations

import random
import string
import sys
import time

try:
    import rs_bpe
except ImportError:
    sys.exit("rs_bpe not available. Build with: maturin develop --release")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_text(approx_tokens: int) -> str:
    """Generate text that produces approximately `approx_tokens` tokens.
    English prose averages ~1.3 tokens per word; we use ~4 chars/word."""
    words_needed = int(approx_tokens / 1.3)
    words = []
    for _ in range(words_needed):
        length = random.randint(2, 8)
        words.append("".join(random.choices(string.ascii_lowercase, k=length)))
    return " ".join(words)


def make_messages(total_tokens: int, num_turns: int = 1) -> list[dict]:
    """Build a multi-turn conversation with approximately `total_tokens` tokens."""
    messages = []
    tokens_per_msg = max(50, total_tokens // (num_turns * 2))  # user + assistant pairs
    for i in range(num_turns):
        messages.append({"role": "user", "content": generate_text(tokens_per_msg)})
        if i < num_turns - 1:
            messages.append({"role": "assistant", "content": generate_text(tokens_per_msg)})
    return messages


def time_fn(fn, *, warmup: int = 1, repeats: int = 5) -> float:
    """Time a function, returning the median time in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

def benchmark_single_turn_cold(sizes: list[int]):
    """Scenario 1: Single-turn cold encoding at various input sizes."""
    print("\n" + "=" * 70)
    print("Scenario 1: Single-turn cold start")
    print("=" * 70)
    print(f"{'Tokens':>10} {'Baseline ms':>12} {'Direct ms':>12} {'Speedup':>8}")
    print("-" * 50)

    tok = rs_bpe.kimi_k2()

    for target_tokens in sizes:
        messages = make_messages(target_tokens, num_turns=1)

        # Baseline: apply_chat_template → encode
        def baseline():
            prompt = tok.apply_chat_template(messages)
            return tok.encode(prompt)

        # Direct emission
        def direct():
            return tok.tokenize_messages_direct(messages)

        t_base = time_fn(baseline)
        t_direct = time_fn(direct)

        # Verify equivalence
        base_tokens = baseline()
        direct_tokens = direct()
        assert base_tokens == direct_tokens, "Token mismatch!"

        actual_tokens = len(base_tokens)
        speedup = t_base / t_direct if t_direct > 0 else float("inf")
        print(f"{actual_tokens:>10,} {t_base:>12.2f} {t_direct:>12.2f} {speedup:>7.2f}x")


def benchmark_multi_turn_warm(base_tokens: int, num_turns: int):
    """Scenario 2/3: Multi-turn with warm cache."""
    print(f"\n{'=' * 70}")
    print(f"Scenario: Multi-turn warm cache ({num_turns} turns, ~{base_tokens//1000}K base)")
    print("=" * 70)
    print(f"{'Turn':>5} {'Msgs':>5} {'Baseline ms':>12} {'Direct ms':>12} "
          f"{'Cached ms':>12} {'Cache Spdup':>12}")
    print("-" * 70)

    tok = rs_bpe.kimi_k2()
    encoder = rs_bpe.KimiChatEncoder()

    # Build initial context
    messages = make_messages(base_tokens, num_turns=1)

    for turn in range(1, num_turns + 1):
        # Baseline
        def baseline():
            prompt = tok.apply_chat_template(messages)
            return tok.encode(prompt)

        # Direct emission
        def direct():
            return tok.tokenize_messages_direct(messages)

        # ChatEncoder (cached)
        def cached():
            return encoder.encode_messages(messages)

        t_base = time_fn(baseline)
        t_direct = time_fn(direct)
        t_cached = time_fn(cached)

        # Verify
        base_tokens_out = baseline()
        cached_tokens = cached()
        assert base_tokens_out == cached_tokens, f"Token mismatch at turn {turn}!"

        speedup_vs_base = t_base / t_cached if t_cached > 0 else float("inf")
        print(f"{turn:>5} {len(messages):>5} {t_base:>12.2f} {t_direct:>12.2f} "
              f"{t_cached:>12.2f} {speedup_vs_base:>11.2f}x")

        # Add new turn
        messages.append({"role": "assistant", "content": generate_text(300)})
        messages.append({"role": "user", "content": generate_text(300)})


def benchmark_cold_start_parallel(sizes: list[int]):
    """Scenario: Cold start with many messages (parallel encoding benefit)."""
    print("\n" + "=" * 70)
    print("Scenario: Cold start (ChatEncoder, all messages uncached)")
    print("=" * 70)
    print(f"{'Tokens':>10} {'Msgs':>5} {'Direct ms':>12} {'Encoder ms':>12} {'Speedup':>8}")
    print("-" * 55)

    tok = rs_bpe.kimi_k2()

    for target_tokens in sizes:
        # Many short messages to benefit from parallel encoding
        num_msgs = max(10, target_tokens // 500)
        messages = make_messages(target_tokens, num_turns=num_msgs // 2)

        def direct():
            return tok.tokenize_messages_direct(messages)

        # Fresh encoder each time (cold cache)
        def encoder_cold():
            enc = rs_bpe.KimiChatEncoder()
            return enc.encode_messages(messages)

        t_direct = time_fn(direct)
        t_encoder = time_fn(encoder_cold)

        # Verify
        d = direct()
        e = encoder_cold()
        assert d == e, "Token mismatch!"

        speedup = t_direct / t_encoder if t_encoder > 0 else float("inf")
        print(f"{len(d):>10,} {len(messages):>5} {t_direct:>12.2f} "
              f"{t_encoder:>12.2f} {speedup:>7.2f}x")


def benchmark_end_to_end_session():
    """Scenario: Simulate a full Cursor session with realistic turn progression."""
    print("\n" + "=" * 70)
    print("Scenario: Full session simulation (10 turns, 75K base context)")
    print("=" * 70)
    print(f"{'Turn':>5} {'Total tok':>10} {'Baseline ms':>12} {'Encoder ms':>12} "
          f"{'Cumul Spdup':>12}")
    print("-" * 55)

    tok = rs_bpe.kimi_k2()
    encoder = rs_bpe.KimiChatEncoder()

    messages = [
        {"role": "system", "content": generate_text(500)},
        {"role": "user", "content": generate_text(70000)},
    ]

    cumul_base = 0.0
    cumul_enc = 0.0

    for turn in range(1, 11):
        def baseline():
            prompt = tok.apply_chat_template(messages)
            return tok.encode(prompt)

        def cached():
            return encoder.encode_messages(messages)

        t_base = time_fn(baseline, warmup=0, repeats=3)
        t_enc = time_fn(cached, warmup=0, repeats=3)

        cumul_base += t_base
        cumul_enc += t_enc

        base_out = baseline()
        cumul_speedup = cumul_base / cumul_enc if cumul_enc > 0 else float("inf")
        print(f"{turn:>5} {len(base_out):>10,} {t_base:>12.2f} {t_enc:>12.2f} "
              f"{cumul_speedup:>11.2f}x")

        # Simulate assistant response + user follow-up
        messages.append({"role": "assistant", "content": generate_text(300)})
        messages.append({"role": "user", "content": generate_text(300)})

    print(f"\nCumulative: baseline={cumul_base:.1f}ms, encoder={cumul_enc:.1f}ms, "
          f"speedup={cumul_base/cumul_enc:.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    print("Kimi K2 Tokenizer Optimization Benchmark")
    print("Cursor-representative workloads\n")

    # Warm up tokenizer
    tok = rs_bpe.kimi_k2()
    tok.encode("warmup")

    # Scenario 1: Single-turn cold at Cursor input sizes
    benchmark_single_turn_cold([15_000, 40_000, 75_000, 115_000, 175_000])

    # Scenario 2: Multi-turn warm (10 turns, 75K base — Cursor median)
    benchmark_multi_turn_warm(75_000, num_turns=10)

    # Scenario 3: Multi-turn long session (30 turns — 3.6% of traffic)
    benchmark_multi_turn_warm(75_000, num_turns=30)

    # Scenario 4: Cold start with parallel encoding benefit
    benchmark_cold_start_parallel([15_000, 40_000, 75_000, 115_000, 175_000])

    # Scenario 5: Full session simulation
    benchmark_end_to_end_session()


if __name__ == "__main__":
    main()
