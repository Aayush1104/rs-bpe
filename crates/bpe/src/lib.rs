//! Byte Pair Encoding (BPE) implementation for efficient text tokenization.
//!
//! This crate provides various implementations of the BPE algorithm for tokenizing text.
//! It includes different encoder strategies and utilities for efficient token counting.
pub mod appendable_encoder;
pub mod backtrack_encoder;
mod bitfield;
pub mod byte_pair_encoding;
pub mod interval_encoding;
pub mod prependable_encoder;
