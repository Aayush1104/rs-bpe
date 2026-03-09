"""
Speed benchmark comparing rs_bpe.kimi_k2() against the official HuggingFace
tokenizer (moonshotai/Kimi-K2-Instruct).

Benchmarks:
  1) Single-text encoding (small / medium / large)
  2) Single-text decoding
  3) Roundtrip (encode + decode)
  4) Batch encoding
"""

from __future__ import annotations

import gc
import random
import statistics
import sys
import time
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

try:
    import rs_bpe
    RS_BPE_AVAILABLE = True
except ImportError:
    print("Warning: rs_bpe not available. Build with: maturin develop --release")
    RS_BPE_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    print("Warning: transformers not available. pip install transformers")
    HF_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

SMALL_TEXT = "Hello, world! This is a small test string for tokenization."
SMALL_TEXT_CJK = "你好世界！这是一段用于分词测试的短文本。"

try:
    with open("README.md") as f:
        MEDIUM_TEXT = f.read()
except FileNotFoundError:
    MEDIUM_TEXT = SMALL_TEXT * 20

LARGE_TEXT = MEDIUM_TEXT * 50

MEDIUM_TEXT_CJK = (
    "人工智能正在改变世界。大语言模型通过海量数据进行训练，"
    "能够理解并生成自然语言。Transformer架构是目前最流行的深度学习模型之一。"
) * 20

TEST_TEXTS = {
    "small": SMALL_TEXT,
    "small_cjk": SMALL_TEXT_CJK,
    "medium": MEDIUM_TEXT,
    "medium_cjk": MEDIUM_TEXT_CJK,
    "large": LARGE_TEXT,
}


def generate_batch_data(
    num_texts: int = 1000,
    text_length_range: tuple[int, int] = (50, 500),
) -> list[str]:
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "Python", "Rust", "encoding", "tokenization", "performance",
        "batch", "processing", "algorithm", "benchmark", "test",
        "人工智能", "深度学习", "自然语言", "模型", "训练",
    ]
    punctuation = [",", ".", "!", "?", ";", ":", "-"]
    rng = random.Random(42)

    result = []
    for _ in range(num_texts):
        length = rng.randint(*text_length_range)
        parts: list[str] = []
        while len(" ".join(parts)) < length:
            parts.append(rng.choice(words))
            if rng.random() < 0.2:
                parts.append(rng.choice(punctuation))
            if rng.random() < 0.05:
                parts.append("\n")
        result.append(" ".join(parts))
    return result


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class TokenizerAdapter(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> str: ...

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]


class KimiHFAdapter(TokenizerAdapter):
    def __init__(self):
        super().__init__("hf_kimi_k2")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "moonshotai/Kimi-K2-Instruct", trust_remote_code=True
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)


class KimiRsBpeAdapter(TokenizerAdapter):
    def __init__(self):
        super().__init__("rs_bpe_kimi_k2")
        self.tokenizer = rs_bpe.kimi_k2()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        result = self.tokenizer.decode(tokens)
        return result if result is not None else ""


class KimiRsBpeBatchAdapter(TokenizerAdapter):
    def __init__(self):
        super().__init__("rs_bpe_kimi_k2_batch")
        self.tokenizer = rs_bpe.kimi_k2()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        result = self.tokenizer.decode(tokens)
        return result if result is not None else ""

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        tokens, _, _ = self.tokenizer.encode_batch(texts)
        return tokens


class KimiRsBpeParallelAdapter(TokenizerAdapter):
    def __init__(self):
        super().__init__("rs_bpe_kimi_k2_parallel")
        self.tokenizer = rs_bpe.kimi_k2()
        self.parallel_options = rs_bpe.ParallelOptions(
            min_batch_size=10, chunk_size=100, max_threads=0
        )
        print(f"  Parallel adapter: {rs_bpe.get_num_threads()} threads available")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int]) -> str:
        result = self.tokenizer.decode(tokens)
        return result if result is not None else ""

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        tokens, _, _, _ = self.tokenizer.encode_batch_parallel(
            texts, self.parallel_options
        )
        return tokens


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def time_func(func, *args, num_runs: int = 5) -> tuple[float, object]:
    gc.collect()
    # warmup
    result = func(*args)
    times = []
    for _ in range(num_runs):
        gc.collect()
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return statistics.mean(times), result


def format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def run_single_benchmarks(adapters: list[TokenizerAdapter], num_runs: int = 5):
    print("\n===== SINGLE TEXT BENCHMARKS =====")

    for size_label, text in TEST_TEXTS.items():
        byte_size = len(text.encode("utf-8"))
        print(f"\n--- {size_label.upper()} ({format_size(byte_size)}) ---")
        print(f"  {'Tokenizer':<30} {'Encode (s)':>12} {'Decode (s)':>12} {'Roundtrip':>12} {'Tokens':>8} {'Tok/s':>12}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*12}")

        for adapter in adapters:
            enc_time, tokens = time_func(adapter.encode, text, num_runs=num_runs)
            token_count = len(tokens)
            dec_time, _ = time_func(adapter.decode, tokens, num_runs=num_runs)
            rt_time = enc_time + dec_time
            tps = token_count / enc_time if enc_time > 0 else 0

            print(
                f"  {adapter.name:<30} {enc_time:>12.6f} {dec_time:>12.6f} "
                f"{rt_time:>12.6f} {token_count:>8} {tps:>12,.0f}"
            )


def run_batch_benchmarks(
    adapters: list[TokenizerAdapter],
    batch_sizes: list[int] | None = None,
    num_runs: int = 3,
):
    if batch_sizes is None:
        batch_sizes = [1, 10, 100, 1000]

    print("\n===== BATCH ENCODING BENCHMARKS =====")
    max_batch = max(batch_sizes)
    test_data = generate_batch_data(num_texts=max_batch + 10)

    for batch_size in batch_sizes:
        texts = test_data[:batch_size]
        print(f"\n--- Batch size: {batch_size} ---")
        print(f"  {'Tokenizer':<30} {'Time (s)':>12} {'Tokens':>10} {'Tok/s':>12}")
        print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*12}")

        for adapter in adapters:
            enc_time, token_batches = time_func(
                adapter.encode_batch, texts, num_runs=num_runs
            )
            total_tokens = sum(len(t) for t in token_batches)
            tps = total_tokens / enc_time if enc_time > 0 else 0

            print(
                f"  {adapter.name:<30} {enc_time:>12.6f} {total_tokens:>10} {tps:>12,.0f}"
            )


def print_speedup_summary(adapters: list[TokenizerAdapter], num_runs: int = 5):
    print("\n===== SPEEDUP SUMMARY (rs_bpe over HF) =====")

    hf_adapter = None
    rs_adapter = None
    for a in adapters:
        if "hf" in a.name:
            hf_adapter = a
        elif a.name == "rs_bpe_kimi_k2":
            rs_adapter = a

    if hf_adapter is None or rs_adapter is None:
        print("  Cannot compute speedup: need both HF and rs_bpe adapters")
        return

    print(f"\n  {'Text':>15} {'HF (s)':>12} {'rs_bpe (s)':>12} {'Speedup':>10}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*10}")

    for size_label, text in TEST_TEXTS.items():
        hf_time, _ = time_func(hf_adapter.encode, text, num_runs=num_runs)
        rs_time, _ = time_func(rs_adapter.encode, text, num_runs=num_runs)
        speedup = hf_time / rs_time if rs_time > 0 else float("inf")
        print(
            f"  {size_label:>15} {hf_time:>12.6f} {rs_time:>12.6f} {speedup:>9.2f}x"
        )


def plot_results(adapters: list[TokenizerAdapter], num_runs: int = 5):
    if not PLOT_AVAILABLE:
        print("\nSkipping plots (matplotlib not available)")
        return

    sorted_sizes = sorted(TEST_TEXTS.keys(), key=lambda k: len(TEST_TEXTS[k]))
    byte_sizes = [len(TEST_TEXTS[s].encode("utf-8")) for s in sorted_sizes]
    labels = [f"{s}\n({format_size(b)})" for s, b in zip(sorted_sizes, byte_sizes)]

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    markers = ["o", "s", "d", "^", "v"]

    # Collect encode times
    encode_data: dict[str, list[float]] = {}
    tps_data: dict[str, list[float]] = {}
    for adapter in adapters:
        enc_times = []
        tps_vals = []
        for size in sorted_sizes:
            t, tokens = time_func(adapter.encode, TEST_TEXTS[size], num_runs=num_runs)
            enc_times.append(t)
            tps_vals.append(len(tokens) / t if t > 0 else 0)
        encode_data[adapter.name] = enc_times
        tps_data[adapter.name] = tps_vals

    # Plot encode time
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(sorted_sizes))
    for i, adapter in enumerate(adapters):
        ax1.plot(
            x, encode_data[adapter.name],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2, markersize=8, label=adapter.name,
        )
    ax1.set_yscale("log")
    ax1.set_title("Kimi K2 Encoding Time (lower is better)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Time (seconds) - Log Scale")
    ax1.set_xlabel("Input Size")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(loc="upper left", fancybox=True, shadow=True)
    fig1.tight_layout()
    fig1.savefig("benchmark/kimi_k2_benchmark_results_time.svg", dpi=300, format="svg", bbox_inches="tight")
    print("\nSaved benchmark/kimi_k2_benchmark_results_time.svg")

    # Plot throughput
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for i, adapter in enumerate(adapters):
        ax2.plot(
            x, tps_data[adapter.name],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2, markersize=8, label=adapter.name,
        )
    ax2.set_title("Kimi K2 Encoding Throughput (higher is better)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Tokens per Second")
    ax2.set_xlabel("Input Size")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc="upper left", fancybox=True, shadow=True)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))
    ax2.set_ylim(bottom=0)
    fig2.tight_layout()
    fig2.savefig("benchmark/kimi_k2_benchmark_results_throughput.svg", dpi=300, format="svg", bbox_inches="tight")
    print("Saved benchmark/kimi_k2_benchmark_results_throughput.svg")

    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Initializing tokenizers...")
    adapters: list[TokenizerAdapter] = []

    if HF_AVAILABLE:
        adapters.append(KimiHFAdapter())
    if RS_BPE_AVAILABLE:
        adapters.append(KimiRsBpeAdapter())
        adapters.append(KimiRsBpeBatchAdapter())
        adapters.append(KimiRsBpeParallelAdapter())

    if not adapters:
        print("Error: No tokenizers available.")
        sys.exit(1)

    num_runs = 5
    run_single_benchmarks(adapters, num_runs=num_runs)

    # Batch benchmarks only for adapters with batch support
    batch_adapters = adapters
    run_batch_benchmarks(batch_adapters, num_runs=3)

    print_speedup_summary(adapters, num_runs=num_runs)
    plot_results(adapters, num_runs=num_runs)

    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
