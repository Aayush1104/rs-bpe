"""
Generate Kimi K2 tiktoken data files for rs-bpe.

Downloads tiktoken.model from moonshotai/Kimi-K2-Instruct on HuggingFace,
verifies it contains base tokens (IDs 0-163583), and creates the gzipped
tiktoken file and special tokens JSON.

Usage:
    python scripts/generate_kimi_k2.py

Requirements:
    pip install huggingface_hub
"""

import gzip
import json
import os
from pathlib import Path


def main():
    # Try to use a local cached copy first, otherwise download
    tiktoken_path = os.environ.get("KIMI_K2_TIKTOKEN_PATH")
    if tiktoken_path is None:
        from huggingface_hub import hf_hub_download

        tiktoken_path = hf_hub_download(
            repo_id="moonshotai/Kimi-K2-Instruct",
            filename="tiktoken.model",
        )

    print(f"Using tiktoken model from: {tiktoken_path}")

    # Read and verify the tiktoken file
    with open(tiktoken_path) as f:
        lines = f.readlines()

    num_base_tokens = len(lines)
    print(f"Base tokens: {num_base_tokens}")
    assert num_base_tokens == 163584, f"Expected 163584 base tokens, got {num_base_tokens}"

    # Verify IDs are 0 to 163583
    for i, line in enumerate(lines):
        parts = line.strip().split("\t")
        assert len(parts) == 2, f"Line {i}: expected 2 tab-separated fields"
        assert int(parts[1]) == i, f"Line {i}: expected ID {i}, got {parts[1]}"

    # Write gzipped tiktoken file
    output_dir = Path(__file__).parent.parent / "crates" / "bpe-openai" / "data"
    gz_path = output_dir / "kimi_k2.tiktoken.gz"
    with gzip.open(gz_path, "wb") as f:
        for line in lines:
            f.write(line.encode())
    print(f"Wrote {gz_path} ({gz_path.stat().st_size} bytes)")

    # Generate special tokens JSON
    # From tokenizer_config.json and tokenization_kimi.py:
    # 256 special tokens starting at ID 163584
    named_tokens = {
        163584: "[BOS]",
        163585: "[EOS]",
        163586: "<|im_end|>",
        163587: "<|im_user|>",
        163588: "<|im_assistant|>",
        163590: "<|start_header_id|>",
        163591: "<|end_header_id|>",
        163593: "[EOT]",
        163594: "<|im_system|>",
        163595: "<|tool_calls_section_begin|>",
        163596: "<|tool_calls_section_end|>",
        163597: "<|tool_call_begin|>",
        163598: "<|tool_call_argument_begin|>",
        163599: "<|tool_call_end|>",
        163601: "<|im_middle|>",
        163838: "[UNK]",
        163839: "[PAD]",
    }

    special = {}
    for i in range(num_base_tokens, num_base_tokens + 256):
        if i in named_tokens:
            special[named_tokens[i]] = i
        else:
            special[f"<|reserved_token_{i}|>"] = i

    json_path = output_dir / "kimi_k2_special.json"
    with open(json_path, "w") as f:
        json.dump(special, f, indent=2, ensure_ascii=False)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
