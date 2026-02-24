use std::ffi::c_char;
use std::slice;

use bpe_openai::Tokenizer;

/// Opaque handle wrapping a reference to the static DeepSeek tokenizer singleton.
/// The inner reference is `&'static Tokenizer`, so this is inherently thread-safe
/// and can be used from multiple goroutines concurrently without any locking.
pub struct RsBpeTokenizer {
    inner: &'static Tokenizer,
}

/// Result of an encode operation. Returned by value across FFI.
#[repr(C)]
pub struct RsBpeEncodeResult {
    /// Pointer to a Rust-allocated array of token IDs (u32).
    /// Caller must free this with `rsbpe_free_tokens`.
    pub tokens: *mut u32,
    /// Number of tokens in the array.
    pub len: usize,
    /// Error code: 0 = success, 1 = invalid UTF-8 input.
    pub error_code: i32,
}

/// Create a new DeepSeek tokenizer handle.
///
/// Returns a heap-allocated `RsBpeTokenizer` wrapping the static singleton.
/// The caller owns the returned pointer and must free it with `rsbpe_free_tokenizer`.
///
/// # Safety
/// The returned pointer is valid until freed with `rsbpe_free_tokenizer`.
#[no_mangle]
pub extern "C" fn rsbpe_new_deepseek() -> *mut RsBpeTokenizer {
    let tok = RsBpeTokenizer {
        inner: bpe_openai::deepseek_base(),
    };
    Box::into_raw(Box::new(tok))
}

/// Encode a UTF-8 text string into token IDs.
///
/// # Parameters
/// - `handle`: Pointer to a tokenizer created by `rsbpe_new_deepseek`.
/// - `text_ptr`: Pointer to UTF-8 encoded text bytes (does not need to be null-terminated).
/// - `text_len`: Length of the text in bytes.
///
/// # Returns
/// An `RsBpeEncodeResult` with token array, length, and error code.
/// On success (error_code == 0), the caller must free the token array with `rsbpe_free_tokens`.
/// On error (error_code != 0), `tokens` is null and `len` is 0.
///
/// # Safety
/// - `handle` must be a valid pointer from `rsbpe_new_deepseek`.
/// - `text_ptr` must point to `text_len` valid bytes.
/// - The text is only borrowed for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn rsbpe_encode(
    handle: *const RsBpeTokenizer,
    text_ptr: *const c_char,
    text_len: usize,
) -> RsBpeEncodeResult {
    if handle.is_null() || text_ptr.is_null() {
        return RsBpeEncodeResult {
            tokens: std::ptr::null_mut(),
            len: 0,
            error_code: 1,
        };
    }

    let tok = &*handle;
    let bytes = slice::from_raw_parts(text_ptr as *const u8, text_len);

    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => {
            return RsBpeEncodeResult {
                tokens: std::ptr::null_mut(),
                len: 0,
                error_code: 1,
            };
        }
    };

    let token_ids: Vec<u32> = tok.inner.encode(text, None);
    let len = token_ids.len();

    if len == 0 {
        return RsBpeEncodeResult {
            tokens: std::ptr::null_mut(),
            len: 0,
            error_code: 0,
        };
    }

    let mut boxed = token_ids.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    std::mem::forget(boxed);

    RsBpeEncodeResult {
        tokens: ptr,
        len,
        error_code: 0,
    }
}

/// Free a token array previously returned by `rsbpe_encode`.
///
/// # Safety
/// - `tokens` must be a pointer returned by `rsbpe_encode`, or null.
/// - `len` must be the length returned by the same `rsbpe_encode` call.
/// - Must only be called once per allocation.
#[no_mangle]
pub unsafe extern "C" fn rsbpe_free_tokens(tokens: *mut u32, len: usize) {
    if !tokens.is_null() && len > 0 {
        let _ = Box::from_raw(slice::from_raw_parts_mut(tokens, len));
    }
}

/// Free a tokenizer handle previously returned by `rsbpe_new_deepseek`.
///
/// # Safety
/// - `handle` must be a pointer returned by `rsbpe_new_deepseek`, or null.
/// - Must only be called once per handle.
#[no_mangle]
pub unsafe extern "C" fn rsbpe_free_tokenizer(handle: *mut RsBpeTokenizer) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}
