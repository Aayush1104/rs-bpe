#!/usr/bin/env python3
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "huggingface-hub",
# ]
# ///

"""
Convert a HuggingFace tokenizer.json to tiktoken format.

Usage:
    python convert_to_tiktoken.py nvidia/DeepSeek-V3-0324-NVFP4 output.tiktoken

Or with a local file:
    python convert_to_tiktoken.py ./tokenizer.json output.tiktoken
"""

import argparse
import base64
import json
from pathlib import Path


def load_tokenizer_json(source: str) -> dict:
    """Load tokenizer.json from the HuggingFace Hub or a local file."""
    if Path(source).exists():
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # Download from HuggingFace Hub
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=source, filename="tokenizer.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def convert_hf_to_tiktoken(
    tokenizer_json: dict, include_special_tokens: bool = False
) -> tuple[list[tuple[bytes, int]], dict[str, int]]:
    """
    Convert a HuggingFace tokenizer.json to tiktoken format.

    Returns:
        (vocab_list, special_tokens_dict)
        - vocab_list: [(token_bytes, rank), ...] sorted by rank
        - special_tokens_dict: {special_token_str: id}

    """
    model = tokenizer_json.get("model", {})
    model_type = model.get("type", "BPE")

    if model_type != "BPE":
        raise ValueError(f"Only BPE tokenizers are supported, got: {model_type}")

    # 1. Extract vocab
    vocab: dict[str, int] = model.get("vocab", {})

    # 2. Extract merges (used to determine rank order)
    merges: list[str] = model.get("merges", [])

    # 3. Extract special tokens
    added_tokens = tokenizer_json.get("added_tokens", [])
    special_tokens: dict[str, int] = {}
    special_token_ids: set = set()

    for token_info in added_tokens:
        if token_info.get("special", False):
            content = token_info["content"]
            token_id = token_info["id"]
            special_tokens[content] = token_id
            special_token_ids.add(token_id)

    # 4. Build rank mapping
    # tiktoken rank determines BPE merge priority
    # smaller rank = higher priority = more frequent token

    # Method: use merge order to determine rank
    # merges earlier in the list get lower ranks

    # First, base byte tokens (0-255) get the highest priority
    token_to_rank: dict[str, int] = {}
    current_rank = 0

    # Base byte tokens
    for i in range(256):
        byte_token = bytes([i])
        # HuggingFace may represent bytes in different ways
        # Try several possible representations
        possible_representations = [
            byte_token.decode("latin-1"),  # direct byte
            f"<0x{i:02X}>",  # <0x00> format
            chr(i) if i < 128 and chr(i).isprintable() else None,
        ]

        for repr in possible_representations:
            if repr and repr in vocab:
                if repr not in token_to_rank:
                    token_to_rank[repr] = current_rank
                    current_rank += 1
                break

    # Then add merged tokens in merge order
    for merge in merges:
        parts = merge.split(" ")
        if len(parts) == 2:
            merged_token = "".join(parts)
            if merged_token in vocab and merged_token not in token_to_rank:
                token_to_rank[merged_token] = current_rank
                current_rank += 1

    # Add remaining vocab tokens (in original ID order)
    remaining = [
        (token, id)
        for token, id in vocab.items()
        if token not in token_to_rank and id not in special_token_ids
    ]
    remaining.sort(key=lambda x: x[1])

    for token, _ in remaining:
        if token not in token_to_rank:
            token_to_rank[token] = current_rank
            current_rank += 1

    # 5. Convert to bytes and build output
    vocab_list: list[tuple[bytes, int]] = []

    for token_str, rank in token_to_rank.items():
        token_id = vocab.get(token_str)
        if token_id is None or token_id in special_token_ids:
            continue

        # Convert token string to bytes
        token_bytes = convert_token_to_bytes(token_str, model)
        if token_bytes is not None:
            vocab_list.append((token_bytes, rank))

    # Sort by rank
    vocab_list.sort(key=lambda x: x[1])

    # Optional: include special tokens
    if include_special_tokens:
        for token_str in special_tokens:
            token_bytes = token_str.encode("utf-8")
            vocab_list.append((token_bytes, len(vocab_list)))

    return vocab_list, special_tokens


def convert_token_to_bytes(token_str: str, model: dict) -> bytes | None:
    """
    Convert a HuggingFace token string to raw bytes.

    HuggingFace BPE tokenizers may use different byte representations:
    1. Direct UTF-8 strings
    2. GPT-2-style byte encoding (Ġ = space, Ċ = newline, etc.)
    3. <0xAB> hex format
    """
    # Check for byte_fallback setting
    byte_fallback = model.get("byte_fallback", False)

    # GPT-2 / cl100k-style byte mapping
    # These models use special characters to represent non-printable bytes
    gpt2_byte_encoder = _get_gpt2_byte_encoder()
    gpt2_byte_decoder = {v: k for k, v in gpt2_byte_encoder.items()}

    try:
        # Try GPT-2-style decoding
        if all(c in gpt2_byte_decoder for c in token_str):
            return bytes([gpt2_byte_decoder[c] for c in token_str])
    except ValueError:
        pass

    # Try <0xAB> format
    if token_str.startswith("<0x") and token_str.endswith(">"):
        try:
            byte_val = int(token_str[3:-1], 16)
            return bytes([byte_val])
        except ValueError:
            pass

    # Direct UTF-8 encoding
    try:
        return token_str.encode("utf-8")
    except UnicodeEncodeError:
        pass

    # Latin-1 fallback
    try:
        return token_str.encode("latin-1")
    except UnicodeEncodeError:
        return None


def _get_gpt2_byte_encoder() -> dict[int, str]:
    """
    GPT-2 byte-to-Unicode mapping.
    Used to map non-printable bytes to printable Unicode characters.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def write_tiktoken_file(
    vocab_list: list[tuple[bytes, int]], output_path: str, compress: bool = False
):
    """
    Write tiktoken format file.

    Format: each line is `<base64_encoded_bytes> <rank>`
    """
    lines = []
    for token_bytes, rank in vocab_list:
        b64 = base64.b64encode(token_bytes).decode("ascii")
        lines.append(f"{b64} {rank}")

    content = "\n".join(lines)

    if compress:
        import gzip

        with gzip.open(output_path + ".gz", "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Written {len(vocab_list)} tokens to {output_path}")


def write_special_tokens_json(special_tokens: dict[str, int], output_path: str):
    """Write special tokens to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens, f, ensure_ascii=False, indent=2)
    print(f"Written {len(special_tokens)} special tokens to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace tokenizer.json to tiktoken format"
    )
    parser.add_argument("source", help="HuggingFace model ID or path to tokenizer.json")
    parser.add_argument("output", help="Output tiktoken file path")
    parser.add_argument(
        "--compress", "-c", action="store_true", help="Compress output with gzip"
    )
    parser.add_argument(
        "--special-tokens", "-s", help="Output path for special tokens JSON"
    )
    parser.add_argument(
        "--include-special",
        action="store_true",
        help="Include special tokens in main vocab file",
    )

    args = parser.parse_args()

    print(f"Loading tokenizer from: {args.source}")
    tokenizer_json = load_tokenizer_json(args.source)

    print("Converting to tiktoken format...")
    vocab_list, special_tokens = convert_hf_to_tiktoken(
        tokenizer_json, include_special_tokens=args.include_special
    )

    write_tiktoken_file(vocab_list, args.output, compress=args.compress)

    if args.special_tokens:
        write_special_tokens_json(special_tokens, args.special_tokens)
    elif special_tokens:
        # Write special tokens by default
        special_path = args.output.rsplit(".", 1)[0] + "_special.json"
        write_special_tokens_json(special_tokens, special_path)

    print("\nConversion complete!")
    print(f"  Vocab size: {len(vocab_list)}")
    print(f"  Special tokens: {len(special_tokens)}")


if __name__ == "__main__":
    main()
