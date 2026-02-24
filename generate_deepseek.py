import base64
import gzip
import json

TOKENIZER_PATH = (
    "/Users/fming/.cache/huggingface/hub/models--nvidia--DeepSeek-V3-0324-NVFP4/"
    "snapshots/d03662cdc34b56eab1315d4557e395e6b4944782/tokenizer.json"
)
OUTPUT_PATH = "bpe-openai/data/deepseek_base.tiktoken.gz"


def bytes_to_unicode() -> dict[str, int]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def vocab_token_to_bytes(token: str, u2b: dict[str, int]) -> bytes:
    if all(ch in u2b for ch in token):
        return bytes(u2b[ch] for ch in token)
    return token.encode("utf-8")


with open(TOKENIZER_PATH) as f:
    tokenizer = json.load(f)

u2b = bytes_to_unicode()
vocab = tokenizer["model"]["vocab"]
added_tokens = tokenizer["added_tokens"]

max_id = max(max(vocab.values()), max(t["id"] for t in added_tokens))
tokens_by_id: list[bytes | None] = [None] * (max_id + 1)

for token, idx in vocab.items():
    tokens_by_id[idx] = vocab_token_to_bytes(token, u2b)

for tok in added_tokens:
    idx = tok["id"]
    if tokens_by_id[idx] is not None:
        continue
    tokens_by_id[idx] = tok["content"].encode("utf-8")

missing = [i for i, tok in enumerate(tokens_by_id) if tok is None]
if missing:
    raise RuntimeError(f"Missing token ids: {missing[:10]}")

with gzip.open(OUTPUT_PATH, "wb") as f:
    for idx, token_bytes in enumerate(tokens_by_id):
        token_b64 = base64.b64encode(token_bytes).decode("utf-8")
        f.write(f"{token_b64}\t{idx}\n".encode())
