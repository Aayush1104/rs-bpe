# Repository Guidelines

## Project Structure & Module Organization
- `crates/bpe`: Core BPE implementation and utilities (Rust).
- `crates/bpe-openai`: Prebuilt tokenizers and regex pretokenizer logic (Rust).
- `crates/python-bpe`: PyO3 Python bindings for the Rust crates.
- `python/rs_bpe`: Python package wrapper (`__init__.py`, `bpe.pyi`, `py.typed`).
- `scripts/`: Utility scripts (e.g., tokenizer conversion).
- `assets/`: Prebuilt tokenizer assets and JSONs.
- `tests/` and `benchmark/`: Test cases and benchmark artifacts.

## Build, Test, and Development Commands
- `cargo check --workspace`  
  Type-check all crates.
- `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check --workspace`  
  Required on Python 3.14 to bypass PyO3 version gating.
- `cargo test --workspace`  
  Run Rust tests for all crates.
- `cargo clippy --workspace`  
  Lint Rust codebase.
- `cargo run -p bpe --features "rand tiktoken" --bin find_hash_factor -- <file.tiktoken>`  
  Compute hash factor for tiktoken data.
- `maturin build`  
  Build Python wheel (uses `pyproject.toml` and `crates/python-bpe/Cargo.toml`).

## Required Checks After Rust Changes
- After any Rust code change, run:
  `cargo check --workspace && cargo clippy --workspace`
  and ensure there are no errors or warnings. Use
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` on Python 3.14 if needed.

## Coding Style & Naming Conventions
- Rust: `cargo fmt` formatting, `snake_case` for functions/vars, `CamelCase` for types.
- Python: 4-space indentation, follow Ruff settings in `ruff.toml`.
- Keep docstrings and comments precise and consistent with existing modules.

## Testing Guidelines
- Rust tests live in crate `tests/` modules and `tests/` directory.
- Python interface is validated via Rust-side tests; add Python tests if behavior changes.
- Prefer deterministic tests for tokenization; include special token cases when applicable.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits (e.g., `feat:`, `fix:`, `chore:`).
- PRs should include: summary, rationale, test commands run, and any compatibility notes.
- Link related issues when applicable; include asset changes when tokenizer data updates.

## Configuration Notes
- PyO3 currently targets Python 3.13; for Python 3.14 use
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` during local builds.
