"""
Parallel encoding benchmark for rs_bpe.kimi_k2().

Adapted from encode_parallel_benchmark.py. Compares:
  1) Sequential batch encode vs parallel batch encode
  2) Direct long-text encode vs split_chunks + parallel encode
  3) Direct long-text encode vs encode_split_chunks_parallel
"""

from __future__ import annotations

import argparse
import gc
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

try:
    import rs_bpe
except ImportError as exc:
    raise SystemExit(
        "rs_bpe is not available. Build it first with `maturin develop --release`."
    ) from exc


WORDS = [
    "tokenization", "throughput", "parallel", "sequential", "database",
    "backend", "optimization", "latency", "scheduler", "cache",
    "regex", "encoding", "segment", "pipeline", "rust",
    "python", "thread", "batch", "allocator", "memory",
    # CJK words for representative Kimi K2 workloads
    "人工智能", "深度学习", "自然语言", "大模型", "训练",
    "推理", "分词", "编码", "解码", "优化",
]

PUNCTUATION = [",", ".", "!", "?", ";", ":", "，", "。", "！", "？"]


@dataclass(frozen=True)
class Stats:
    mean: float
    median: float
    minimum: float
    maximum: float
    stdev: float

    @classmethod
    def from_times(cls, times: list[float]) -> Stats:
        if not times:
            raise ValueError("times must not be empty")
        spread = 0.0 if len(times) == 1 else statistics.stdev(times)
        return cls(
            mean=statistics.mean(times),
            median=statistics.median(times),
            minimum=min(times),
            maximum=max(times),
            stdev=spread,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sequential vs parallel batch encode for kimi_k2, "
            "and direct long encode vs split_chunks + parallel encode."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--min-text-len", type=int, default=80)
    parser.add_argument("--max-text-len", type=int, default=240)
    parser.add_argument("--long-base-size", type=int, default=256)
    parser.add_argument("--long-repeat", type=int, default=40)
    parser.add_argument("--split-chunk-size", type=int, default=1024)
    parser.add_argument("--parallel-min-batch-size", type=int, default=20)
    parser.add_argument("--parallel-chunk-size", type=int, default=100)
    parser.add_argument("--parallel-max-threads", type=int, default=0)
    parser.set_defaults(use_thread_pool=True)
    parser.add_argument("--use-thread-pool", action="store_true", dest="use_thread_pool")
    parser.add_argument("--no-thread-pool", action="store_false", dest="use_thread_pool")
    return parser.parse_args()


def generate_texts(
    rng: random.Random, count: int, min_len: int, max_len: int,
) -> list[str]:
    texts: list[str] = []
    for _ in range(count):
        target_len = rng.randint(min_len, max_len)
        words: list[str] = []
        current_len = 0
        while current_len < target_len:
            token = rng.choice(WORDS)
            words.append(token)
            current_len += len(token) + 1
            if rng.random() < 0.2:
                mark = rng.choice(PUNCTUATION)
                words.append(mark)
                current_len += len(mark) + 1
            if rng.random() < 0.03:
                words.append("\n")
                current_len += 1
        texts.append(" ".join(words))
    return texts


def build_parallel_options(args: argparse.Namespace):
    return rs_bpe.ParallelOptions(
        min_batch_size=args.parallel_min_batch_size,
        chunk_size=args.parallel_chunk_size,
        max_threads=args.parallel_max_threads,
        use_thread_pool=args.use_thread_pool,
    )


def run_timed(
    func: Callable[[], int], runs: int, warmups: int,
) -> tuple[Stats, int]:
    for _ in range(warmups):
        _ = func()

    times: list[float] = []
    last_value = 0
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(runs):
            started = time.perf_counter()
            last_value = func()
            elapsed = time.perf_counter() - started
            times.append(elapsed)
    finally:
        if gc_was_enabled:
            gc.enable()

    return Stats.from_times(times), last_value


def flatten_token_batches(token_batches: list[list[int]]) -> list[int]:
    flat: list[int] = []
    for tokens in token_batches:
        flat.extend(tokens)
    return flat


def print_stats(title: str, stats: Stats) -> None:
    print(title)
    print(f"  mean   : {stats.mean:.6f}s")
    print(f"  median : {stats.median:.6f}s")
    print(f"  min/max: {stats.minimum:.6f}s / {stats.maximum:.6f}s")
    print(f"  stdev  : {stats.stdev:.6f}s")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    tokenizer = rs_bpe.kimi_k2()
    parallel_options = build_parallel_options(args)

    texts = generate_texts(
        rng=rng, count=args.batch_size,
        min_len=args.min_text_len, max_len=args.max_text_len,
    )

    seq_tokens, seq_total_tokens, _ = tokenizer.encode_batch(texts)
    par_tokens, par_total_tokens, _, threads_used = tokenizer.encode_batch_parallel(
        texts, parallel_options
    )

    same_batch_result = (
        seq_tokens == par_tokens and seq_total_tokens == par_total_tokens
    )

    print("=== Configuration ===")
    print(
        f"batch_size={args.batch_size}, text_len=[{args.min_text_len}, {args.max_text_len}], "
        f"runs={args.runs}, warmups={args.warmups}"
    )
    print(
        "parallel_options="
        f"(min_batch_size={args.parallel_min_batch_size}, "
        f"chunk_size={args.parallel_chunk_size}, "
        f"max_threads={args.parallel_max_threads}, "
        f"use_thread_pool={args.use_thread_pool})"
    )
    print(f"reported_threads={threads_used}")
    print(f"batch_result_equal={same_batch_result}")
    print()

    # --- Benchmark 1: Sequential vs Parallel batch ---

    def run_sequential_batch() -> int:
        _, total_tokens, _ = tokenizer.encode_batch(texts)
        return total_tokens

    def run_parallel_batch() -> int:
        _, total_tokens, _, _ = tokenizer.encode_batch_parallel(texts, parallel_options)
        return total_tokens

    seq_stats, seq_last = run_timed(run_sequential_batch, runs=args.runs, warmups=args.warmups)
    par_stats, par_last = run_timed(run_parallel_batch, runs=args.runs, warmups=args.warmups)

    speedup_batch = seq_stats.mean / par_stats.mean if par_stats.mean > 0 else float("inf")
    print("=== 1) Sequential Batch vs Parallel Batch ===")
    print(f"sequential_tokens={seq_last}, parallel_tokens={par_last}")
    print_stats("sequential_batch", seq_stats)
    print_stats("parallel_batch  ", par_stats)
    print(f"speedup(parallel over sequential): {speedup_batch:.3f}x")
    print()

    # --- Benchmark 2: Long text: direct vs split+parallel ---

    long_base = " ".join(
        generate_texts(
            rng=rng, count=args.long_base_size,
            min_len=args.min_text_len, max_len=args.max_text_len,
        )
    )
    long_text = " ".join([long_base] * args.long_repeat)

    direct_tokens = tokenizer.encode(long_text)
    pre_split_chunks = tokenizer.split_chunks(long_text, args.split_chunk_size)
    split_batch_tokens, split_total_tokens, _, _ = tokenizer.encode_batch_parallel(
        pre_split_chunks, parallel_options
    )
    merged_split_tokens = flatten_token_batches(split_batch_tokens)
    split_pipeline_result_equal = direct_tokens == merged_split_tokens

    split_api_batch_tokens, split_api_total_tokens, _, _ = (
        tokenizer.encode_split_chunks_parallel(
            long_text, args.split_chunk_size, parallel_options
        )
    )
    merged_split_api_tokens = flatten_token_batches(split_api_batch_tokens)
    split_api_result_equal = direct_tokens == merged_split_api_tokens

    if len(pre_split_chunks) < args.parallel_min_batch_size:
        print(
            "warning: split chunk count is below min_batch_size, "
            "parallel path may fallback to sequential."
        )

    def run_direct_long_encode() -> int:
        return len(tokenizer.encode(long_text))

    def run_split_and_parallel_total() -> int:
        chunks = tokenizer.split_chunks(long_text, args.split_chunk_size)
        _, total_tokens, _, _ = tokenizer.encode_batch_parallel(chunks, parallel_options)
        return total_tokens

    def run_parallel_on_pre_split_chunks() -> int:
        _, total_tokens, _, _ = tokenizer.encode_batch_parallel(
            pre_split_chunks, parallel_options
        )
        return total_tokens

    def run_encode_split_chunks_parallel() -> int:
        _, total_tokens, _, _ = tokenizer.encode_split_chunks_parallel(
            long_text, args.split_chunk_size, parallel_options
        )
        return total_tokens

    direct_stats, direct_last = run_timed(
        run_direct_long_encode, runs=args.runs, warmups=args.warmups
    )
    split_total_stats, split_total_last = run_timed(
        run_split_and_parallel_total, runs=args.runs, warmups=args.warmups
    )
    split_encode_only_stats, split_encode_only_last = run_timed(
        run_parallel_on_pre_split_chunks, runs=args.runs, warmups=args.warmups
    )
    split_api_stats, split_api_last = run_timed(
        run_encode_split_chunks_parallel, runs=args.runs, warmups=args.warmups
    )

    speedup_split_total = (
        direct_stats.mean / split_total_stats.mean
        if split_total_stats.mean > 0 else float("inf")
    )
    speedup_split_encode_only = (
        direct_stats.mean / split_encode_only_stats.mean
        if split_encode_only_stats.mean > 0 else float("inf")
    )
    speedup_split_api = (
        direct_stats.mean / split_api_stats.mean
        if split_api_stats.mean > 0 else float("inf")
    )

    print(
        "=== 2) Long Text: Direct vs split_chunks+parallel vs "
        "encode_split_chunks_parallel ==="
    )
    print(
        f"long_text_chars={len(long_text)}, pre_split_chunks={len(pre_split_chunks)}, "
        f"split_chunk_size={args.split_chunk_size}"
    )
    print(
        "token_count_check="
        f"(direct={len(direct_tokens)}, split_pipeline={len(merged_split_tokens)}, "
        f"split_api={len(merged_split_api_tokens)})"
    )
    print(
        f"long_result_equal=(split_pipeline={split_pipeline_result_equal}, "
        f"split_api={split_api_result_equal})"
    )
    print_stats("direct_encode             ", direct_stats)
    print_stats("split_chunks+parallel     ", split_total_stats)
    print_stats("parallel_on_pre_split_only", split_encode_only_stats)
    print_stats("encode_split_chunks_parallel", split_api_stats)
    print(f"speedup(split+parallel total over direct): {speedup_split_total:.3f}x")
    print(
        "speedup(pre-split parallel only over direct): "
        f"{speedup_split_encode_only:.3f}x"
    )
    print(
        f"speedup(encode_split_chunks_parallel over direct): {speedup_split_api:.3f}x"
    )
    print(
        f"result_tokens=(direct={direct_last}, split_total={split_total_last}, "
        f"pre_split_parallel={split_encode_only_last}, split_api={split_api_last})"
    )
    print(
        "token_count_consistency="
        f"(split_total={split_total_tokens}, split_api={split_api_total_tokens})"
    )


if __name__ == "__main__":
    main()
