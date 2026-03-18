use std::ffi::c_char;
use std::slice;

use bpe_openai::kimi_k2 as kimi_k2_mod;
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

/// Create a new Kimi K2 tokenizer handle.
///
/// Returns a heap-allocated `RsBpeTokenizer` wrapping the static singleton.
/// The caller owns the returned pointer and must free it with `rsbpe_free_tokenizer`.
///
/// # Safety
/// The returned pointer is valid until freed with `rsbpe_free_tokenizer`.
#[no_mangle]
pub extern "C" fn rsbpe_new_kimi_k2() -> *mut RsBpeTokenizer {
    let tok = RsBpeTokenizer {
        inner: bpe_openai::kimi_k2(),
    };
    Box::into_raw(Box::new(tok))
}

/// Encode a UTF-8 text string into token IDs.
///
/// # Parameters
/// - `handle`: Pointer to a tokenizer created by `rsbpe_new_deepseek` or `rsbpe_new_kimi_k2`.
/// - `text_ptr`: Pointer to UTF-8 encoded text bytes (does not need to be null-terminated).
/// - `text_len`: Length of the text in bytes.
///
/// # Returns
/// An `RsBpeEncodeResult` with token array, length, and error code.
/// On success (error_code == 0), the caller must free the token array with `rsbpe_free_tokens`.
/// On error (error_code != 0), `tokens` is null and `len` is 0.
///
/// # Safety
/// - `handle` must be a valid pointer from `rsbpe_new_deepseek` or `rsbpe_new_kimi_k2`.
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

/// Free a tokenizer handle previously returned by `rsbpe_new_deepseek` or `rsbpe_new_kimi_k2`.
///
/// # Safety
/// - `handle` must be a pointer returned by `rsbpe_new_deepseek` or `rsbpe_new_kimi_k2`, or null.
/// - Must only be called once per handle.
#[no_mangle]
pub unsafe extern "C" fn rsbpe_free_tokenizer(handle: *mut RsBpeTokenizer) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle);
    }
}

// ---------------------------------------------------------------------------
// Kimi K2 optimized paths: direct emission + ChatEncoder
// ---------------------------------------------------------------------------

/// Tokenize a JSON-encoded array of messages using the direct emission path.
///
/// # Parameters
/// - `json_ptr`: Pointer to a UTF-8 JSON string encoding `Vec<kimi_k2::Message>`.
/// - `json_len`: Length of the JSON string in bytes.
/// - `add_generation_prompt`: Non-zero to append the generation prompt suffix.
///
/// # Returns
/// `RsBpeEncodeResult` with error_code: 0=success, 2=JSON parse error, 3=unsupported role.
///
/// # Safety
/// - `json_ptr` must point to `json_len` valid bytes.
#[no_mangle]
pub unsafe extern "C" fn rsbpe_tokenize_messages_direct(
    json_ptr: *const c_char,
    json_len: usize,
    add_generation_prompt: i32,
) -> RsBpeEncodeResult {
    if json_ptr.is_null() {
        return RsBpeEncodeResult {
            tokens: std::ptr::null_mut(),
            len: 0,
            error_code: 2,
        };
    }

    let bytes = slice::from_raw_parts(json_ptr as *const u8, json_len);
    let messages: Vec<kimi_k2_mod::Message> = match serde_json::from_slice(bytes) {
        Ok(m) => m,
        Err(_) => {
            return RsBpeEncodeResult {
                tokens: std::ptr::null_mut(),
                len: 0,
                error_code: 2,
            };
        }
    };

    let result = kimi_k2_mod::tokenize_messages_direct(&messages, None, add_generation_prompt != 0);
    match result {
        Ok(token_ids) => {
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
        Err(_) => RsBpeEncodeResult {
            tokens: std::ptr::null_mut(),
            len: 0,
            error_code: 3,
        },
    }
}

/// Opaque wrapper around `kimi_k2::ChatEncoder`.
pub struct RsBpeChatEncoder {
    inner: kimi_k2_mod::ChatEncoder,
}

/// Create a new ChatEncoder for Kimi K2.
///
/// # Safety
/// The returned pointer is valid until freed with `rsbpe_chat_encoder_free`.
#[no_mangle]
pub extern "C" fn rsbpe_chat_encoder_new() -> *mut RsBpeChatEncoder {
    let enc = RsBpeChatEncoder {
        inner: kimi_k2_mod::ChatEncoder::new(),
    };
    Box::into_raw(Box::new(enc))
}

/// Encode a JSON-encoded array of messages using the ChatEncoder.
///
/// # Parameters
/// - `encoder`: Pointer from `rsbpe_chat_encoder_new`.
/// - `json_ptr`: Pointer to a UTF-8 JSON string encoding `Vec<kimi_k2::Message>`.
/// - `json_len`: Length of the JSON string in bytes.
/// - `add_generation_prompt`: Non-zero to append the generation prompt suffix.
///
/// # Returns
/// `RsBpeEncodeResult` with error_code: 0=success, 2=JSON parse error, 3=unsupported role.
///
/// # Safety
/// - `encoder` must be a valid pointer from `rsbpe_chat_encoder_new`.
/// - `json_ptr` must point to `json_len` valid bytes.
/// - This function is NOT thread-safe for the same encoder instance.
#[no_mangle]
pub unsafe extern "C" fn rsbpe_chat_encoder_encode(
    encoder: *mut RsBpeChatEncoder,
    json_ptr: *const c_char,
    json_len: usize,
    add_generation_prompt: i32,
) -> RsBpeEncodeResult {
    if encoder.is_null() || json_ptr.is_null() {
        return RsBpeEncodeResult {
            tokens: std::ptr::null_mut(),
            len: 0,
            error_code: 2,
        };
    }

    let enc = &mut *encoder;
    let bytes = slice::from_raw_parts(json_ptr as *const u8, json_len);
    let messages: Vec<kimi_k2_mod::Message> = match serde_json::from_slice(bytes) {
        Ok(m) => m,
        Err(_) => {
            return RsBpeEncodeResult {
                tokens: std::ptr::null_mut(),
                len: 0,
                error_code: 2,
            };
        }
    };

    let result = enc
        .inner
        .encode_messages(&messages, None, add_generation_prompt != 0);
    match result {
        Ok(token_ids) => {
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
        Err(_) => RsBpeEncodeResult {
            tokens: std::ptr::null_mut(),
            len: 0,
            error_code: 3,
        },
    }
}

/// Return the number of cached message entries in the encoder.
///
/// # Safety
/// `encoder` must be a valid pointer from `rsbpe_chat_encoder_new`.
#[no_mangle]
pub unsafe extern "C" fn rsbpe_chat_encoder_cache_len(
    encoder: *const RsBpeChatEncoder,
) -> usize {
    if encoder.is_null() {
        return 0;
    }
    (*encoder).inner.cache_len()
}

/// Clear the message cache in the encoder.
///
/// # Safety
/// `encoder` must be a valid pointer from `rsbpe_chat_encoder_new`.
#[no_mangle]
pub unsafe extern "C" fn rsbpe_chat_encoder_clear_cache(encoder: *mut RsBpeChatEncoder) {
    if !encoder.is_null() {
        (*encoder).inner.clear_cache();
    }
}

/// Free a ChatEncoder previously created by `rsbpe_chat_encoder_new`.
///
/// # Safety
/// - `encoder` must be a pointer from `rsbpe_chat_encoder_new`, or null.
/// - Must only be called once per encoder.
#[no_mangle]
pub unsafe extern "C" fn rsbpe_chat_encoder_free(encoder: *mut RsBpeChatEncoder) {
    if !encoder.is_null() {
        let _ = Box::from_raw(encoder);
    }
}
