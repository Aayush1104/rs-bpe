#!/usr/bin/env python3

# Test direct imports in the style requested
print("Testing direct imports...")
import rs_bpe

tok = rs_bpe.cl100k_base()

# Test encoding
enc = tok.encode("Hello, world!")
print(f"Encoded tokens: {enc}")

# Test counting
cnt = tok.count("Hello, world!")
print(f"Token count: {cnt}")

# Test decoding
dec = tok.decode(enc)
print(f"Decoded text: {dec}")

print("\nAll tests completed successfully!")
