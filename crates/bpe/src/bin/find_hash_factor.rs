#[cfg(all(feature = "rand", feature = "tiktoken"))]
use std::env;
#[cfg(all(feature = "rand", feature = "tiktoken"))]
use std::fs;

#[cfg(all(feature = "rand", feature = "tiktoken"))]
use bpe::byte_pair_encoding::find_hash_factor_for_tiktoken;

#[cfg(all(feature = "rand", feature = "tiktoken"))]
fn main() {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "find_hash_factor".to_string());
    let path = match args.next() {
        Some(path) => path,
        None => {
            eprintln!("Usage: {program} <tiktoken_file>");
            std::process::exit(2);
        }
    };

    let data = match fs::read_to_string(&path) {
        Ok(data) => data,
        Err(err) => {
            eprintln!("Failed to read {path}: {err}");
            std::process::exit(1);
        }
    };

    let factor = match find_hash_factor_for_tiktoken(&data) {
        Ok(factor) => factor,
        Err(err) => {
            eprintln!("Failed to compute hash factor: {err}");
            std::process::exit(1);
        }
    };

    println!("{factor}");
}

#[cfg(not(all(feature = "rand", feature = "tiktoken")))]
fn main() {
    eprintln!("This binary requires features: rand, tiktoken");
    std::process::exit(1);
}
