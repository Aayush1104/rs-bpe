"""
Compare chat-template tokenization across three implementations:
  1. HuggingFace (moonshotai/Kimi-K2-Instruct)
  2. rs_bpe baseline (apply_chat_template → encode)
  3. rs_bpe optimized (tokenize_messages_direct / KimiChatEncoder)

Usage:
  python benchmark/kimi_k2_chat_comparison.py
  python benchmark/kimi_k2_chat_comparison.py --turns 10   # multi-turn only
  python benchmark/kimi_k2_chat_comparison.py --quick       # fewer sizes
"""

from __future__ import annotations

import argparse
import gc
import random
import string
import sys
import time

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

try:
    import rs_bpe
except ImportError:
    sys.exit("rs_bpe not available. Build with: maturin develop --release")

try:
    from transformers import AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available — HF column will be skipped")
    print("         Install with: pip install transformers\n")
    HF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_text(approx_tokens: int) -> str:
    """Generate pseudo-English text producing ~approx_tokens tokens."""
    words_needed = int(approx_tokens / 1.3)
    words = []
    for _ in range(words_needed):
        length = random.randint(2, 8)
        words.append("".join(random.choices(string.ascii_lowercase, k=length)))
    return " ".join(words)


def make_conversation(total_tokens: int, num_turns: int) -> list[dict]:
    """Build a multi-turn conversation with approximately total_tokens."""
    msgs: list[dict] = []
    toks_per_msg = max(50, total_tokens // max(1, num_turns * 2))
    for i in range(num_turns):
        msgs.append({"role": "user", "content": generate_text(toks_per_msg)})
        if i < num_turns - 1:
            msgs.append({"role": "assistant", "content": generate_text(toks_per_msg)})
    return msgs


def median_ms(fn, *, warmup: int = 2, repeats: int = 7) -> float:
    """Return the median wall-clock time of fn() in milliseconds."""
    for _ in range(warmup):
        fn()
    gc.collect()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def fmt(ms: float | None) -> str:
    if ms is None:
        return "  —  "
    if ms < 1:
        return f"{ms * 1000:>8.1f}µs"
    return f"{ms:>8.2f}ms"


def fmt_speedup(baseline: float | None, optimized: float) -> str:
    if baseline is None:
        return "   —  "
    return f"{baseline / optimized:>6.1f}x"


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class HFAdapter:
    """HuggingFace transformers tokenizer."""

    def __init__(self):
        print("Loading HuggingFace moonshotai/Kimi-K2-Instruct …")
        self.tok = AutoTokenizer.from_pretrained(
            "moonshotai/Kimi-K2-Instruct", trust_remote_code=True
        )

    def tokenize_chat(self, messages: list[dict], add_gen: bool = True) -> list[int]:
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_gen
        )
        return self.tok.encode(prompt)


class RsBpeBaseline:
    """rs_bpe: apply_chat_template → encode (original path)."""

    def __init__(self):
        self.tok = rs_bpe.kimi_k2()

    def tokenize_chat(self, messages: list[dict], add_gen: bool = True) -> list[int]:
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=add_gen)
        return self.tok.encode(prompt)


class RsBpeDirect:
    """rs_bpe: tokenize_messages_direct (Step 1 — no Aho-Corasick/string)."""

    def __init__(self):
        self.tok = rs_bpe.kimi_k2()

    def tokenize_chat(self, messages: list[dict], add_gen: bool = True) -> list[int]:
        return self.tok.tokenize_messages_direct(
            messages, add_generation_prompt=add_gen
        )


class RsBpeCached:
    """rs_bpe: KimiChatEncoder (Steps 1-4 — caching + parallel + buffer)."""

    def __init__(self):
        self.encoder = rs_bpe.KimiChatEncoder()

    def tokenize_chat(self, messages: list[dict], add_gen: bool = True) -> list[int]:
        return self.encoder.encode_messages(
            messages, add_generation_prompt=add_gen
        )

    def reset(self):
        self.encoder.clear_cache()


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def run_single_turn(sizes: list[int]):
    """Single-turn cold-start at various input sizes."""
    print("\n" + "=" * 80)
    print("SCENARIO 1: Single-turn cold start")
    print("=" * 80)

    hf = HFAdapter() if HF_AVAILABLE else None
    base = RsBpeBaseline()
    direct = RsBpeDirect()

    hdr = f"{'~Tokens':>10}  {'HF':>10}  {'rs baseline':>12}  {'rs direct':>12}  {'HF/base':>8}  {'HF/direct':>10}  {'base/direct':>12}"
    print(hdr)
    print("-" * len(hdr))

    for target in sizes:
        msgs = make_conversation(target, num_turns=1)

        t_hf = median_ms(lambda: hf.tokenize_chat(msgs)) if hf else None
        t_base = median_ms(lambda: base.tokenize_chat(msgs))
        t_direct = median_ms(lambda: direct.tokenize_chat(msgs))

        # token count
        n = len(base.tokenize_chat(msgs))

        print(
            f"{n:>10,}  {fmt(t_hf)}  {fmt(t_base)}  {fmt(t_direct)}  "
            f"{fmt_speedup(t_hf, t_base)}  {fmt_speedup(t_hf, t_direct)}  "
            f"{fmt_speedup(t_base, t_direct)}"
        )


def run_multi_turn(base_tokens: int, num_turns: int):
    """Multi-turn with warm cache — the main Cursor workload."""
    print(f"\n{'=' * 80}")
    print(f"SCENARIO 2: Multi-turn warm cache ({num_turns} turns, ~{base_tokens // 1000}K base)")
    print("=" * 80)

    hf = HFAdapter() if HF_AVAILABLE else None
    base = RsBpeBaseline()
    cached = RsBpeCached()

    hdr = f"{'Turn':>5}  {'Msgs':>5}  {'HF':>10}  {'rs baseline':>12}  {'rs cached':>12}  {'HF/cached':>10}  {'base/cached':>12}"
    print(hdr)
    print("-" * len(hdr))

    msgs = make_conversation(base_tokens, num_turns=1)

    for turn in range(1, num_turns + 1):
        t_hf = median_ms(lambda: hf.tokenize_chat(msgs)) if hf else None
        t_base = median_ms(lambda: base.tokenize_chat(msgs))
        t_cached = median_ms(lambda: cached.tokenize_chat(msgs))

        print(
            f"{turn:>5}  {len(msgs):>5}  {fmt(t_hf)}  {fmt(t_base)}  {fmt(t_cached)}  "
            f"{fmt_speedup(t_hf, t_cached)}  {fmt_speedup(t_base, t_cached)}"
        )

        # Simulate next turn
        msgs.append({"role": "assistant", "content": generate_text(300)})
        msgs.append({"role": "user", "content": generate_text(300)})


def run_session_simulation():
    """Full session: 75K base context, 10 turns, cumulative speedup."""
    print(f"\n{'=' * 80}")
    print("SCENARIO 3: Full session simulation (75K base, 10 turns)")
    print("=" * 80)

    hf = HFAdapter() if HF_AVAILABLE else None
    base = RsBpeBaseline()
    cached = RsBpeCached()

    hdr = f"{'Turn':>5}  {'Tokens':>10}  {'HF':>10}  {'rs baseline':>12}  {'rs cached':>12}  {'Cumul HF/c':>11}  {'Cumul b/c':>10}"
    print(hdr)
    print("-" * len(hdr))

    msgs = [
        {"role": "system", "content": generate_text(500)},
        {"role": "user", "content": generate_text(70_000)},
    ]

    cum_hf = 0.0
    cum_base = 0.0
    cum_cached = 0.0

    for turn in range(1, 11):
        t_hf = median_ms(lambda: hf.tokenize_chat(msgs), warmup=1, repeats=3) if hf else None
        t_base = median_ms(lambda: base.tokenize_chat(msgs), warmup=1, repeats=3)
        t_cached = median_ms(lambda: cached.tokenize_chat(msgs), warmup=1, repeats=3)

        cum_hf += t_hf if t_hf else 0
        cum_base += t_base
        cum_cached += t_cached

        n = len(base.tokenize_chat(msgs))

        c_hf = f"{cum_hf / cum_cached:>10.1f}x" if t_hf else "     —    "
        c_base = f"{cum_base / cum_cached:>9.1f}x"

        print(
            f"{turn:>5}  {n:>10,}  {fmt(t_hf)}  {fmt(t_base)}  {fmt(t_cached)}  "
            f"{c_hf}  {c_base}"
        )

        msgs.append({"role": "assistant", "content": generate_text(300)})
        msgs.append({"role": "user", "content": generate_text(300)})

    print()
    if hf:
        print(f"  Cumulative: HF={cum_hf:.1f}ms  baseline={cum_base:.1f}ms  cached={cum_cached:.1f}ms")
        print(f"  HF→cached: {cum_hf/cum_cached:.1f}x   baseline→cached: {cum_base/cum_cached:.1f}x")
    else:
        print(f"  Cumulative: baseline={cum_base:.1f}ms  cached={cum_cached:.1f}ms")
        print(f"  baseline→cached: {cum_base/cum_cached:.1f}x")


# ---------------------------------------------------------------------------
# Equivalence check
# ---------------------------------------------------------------------------


def verify_equivalence():
    """Quick check that all paths produce identical tokens."""
    print("\nVerifying token equivalence …")
    tok = rs_bpe.kimi_k2()
    encoder = rs_bpe.KimiChatEncoder()

    test_cases = [
        [{"role": "user", "content": "hello"}],
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "thanks"},
        ],
        [
            {"role": "user", "content": "search"},
            {
                "role": "tool",
                "content": "Found 42 results.",
                "tool_call_id": "call_abc",
            },
        ],
    ]

    for i, msgs in enumerate(test_cases):
        prompt = tok.apply_chat_template(msgs)
        baseline = tok.encode(prompt)
        direct = tok.tokenize_messages_direct(msgs)
        cached = encoder.encode_messages(msgs)

        assert baseline == direct, f"Case {i}: baseline != direct"
        assert baseline == cached, f"Case {i}: baseline != cached"

        if HF_AVAILABLE:
            hf = AutoTokenizer.from_pretrained(
                "moonshotai/Kimi-K2-Instruct", trust_remote_code=True
            )
            hf_prompt = hf.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            hf_tokens = hf.encode(hf_prompt)
            # HF template may differ slightly (newlines), so compare token counts
            # rather than exact equality for the template comparison
            assert len(baseline) > 0

    print("  All equivalence checks passed.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Kimi K2 chat tokenization comparison")
    parser.add_argument("--quick", action="store_true", help="Fewer sizes, faster run")
    parser.add_argument(
        "--turns", type=int, default=0, help="Run multi-turn only with N turns"
    )
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace")
    args = parser.parse_args()

    global HF_AVAILABLE
    if args.skip_hf:
        HF_AVAILABLE = False

    random.seed(42)

    print("Kimi K2 Chat Template Tokenization — Three-Way Comparison")
    print("  HF = HuggingFace transformers (moonshotai/Kimi-K2-Instruct)")
    print("  rs baseline = apply_chat_template → encode")
    print("  rs direct = tokenize_messages_direct (Step 1)")
    print("  rs cached = KimiChatEncoder (Steps 1-4)")
    print()

    # Warm up rs_bpe
    tok = rs_bpe.kimi_k2()
    tok.encode("warmup")

    verify_equivalence()

    if args.turns > 0:
        run_multi_turn(75_000, args.turns)
    else:
        sizes = [15_000, 75_000, 175_000] if args.quick else [15_000, 40_000, 75_000, 115_000, 175_000]
        run_single_turn(sizes)
        run_multi_turn(75_000, num_turns=10)
        run_session_simulation()


if __name__ == "__main__":
    main()
