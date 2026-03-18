#ifndef RSBPE_H
#define RSBPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tokenizer handle. */
typedef struct RsBpeTokenizer RsBpeTokenizer;

/* Opaque ChatEncoder handle. */
typedef struct RsBpeChatEncoder RsBpeChatEncoder;

/* Result of an encode operation. */
typedef struct {
    uint32_t *tokens;   /* Rust-allocated token array; free with rsbpe_free_tokens */
    size_t    len;      /* Number of tokens */
    int32_t   error_code; /* 0 = success, 1 = invalid UTF-8, 2 = JSON parse error, 3 = unsupported role */
} RsBpeEncodeResult;

/* Create a DeepSeek tokenizer. Caller owns the returned pointer. */
RsBpeTokenizer *rsbpe_new_deepseek(void);

/* Create a Kimi K2 tokenizer. Caller owns the returned pointer. */
RsBpeTokenizer *rsbpe_new_kimi_k2(void);

/* Encode UTF-8 text into token IDs. text_ptr need not be null-terminated. */
RsBpeEncodeResult rsbpe_encode(const RsBpeTokenizer *handle,
                               const char *text_ptr,
                               size_t text_len);

/* Free a token array returned by rsbpe_encode or other encode functions. */
void rsbpe_free_tokens(uint32_t *tokens, size_t len);

/* Free a tokenizer handle returned by rsbpe_new_deepseek or rsbpe_new_kimi_k2. */
void rsbpe_free_tokenizer(RsBpeTokenizer *handle);

/* --- Kimi K2 optimized paths --- */

/* Tokenize a JSON-encoded message array using direct token emission.
   json_ptr: UTF-8 JSON encoding Vec<Message>; need not be null-terminated.
   add_generation_prompt: non-zero to append generation prompt. */
RsBpeEncodeResult rsbpe_tokenize_messages_direct(const char *json_ptr,
                                                  size_t json_len,
                                                  int32_t add_generation_prompt);

/* Create a new ChatEncoder. Caller owns the returned pointer. */
RsBpeChatEncoder *rsbpe_chat_encoder_new(void);

/* Encode messages using a ChatEncoder (with caching + parallel encoding).
   json_ptr: UTF-8 JSON encoding Vec<Message>; need not be null-terminated.
   add_generation_prompt: non-zero to append generation prompt. */
RsBpeEncodeResult rsbpe_chat_encoder_encode(RsBpeChatEncoder *encoder,
                                             const char *json_ptr,
                                             size_t json_len,
                                             int32_t add_generation_prompt);

/* Return the number of cached message entries. */
size_t rsbpe_chat_encoder_cache_len(const RsBpeChatEncoder *encoder);

/* Clear the message cache. */
void rsbpe_chat_encoder_clear_cache(RsBpeChatEncoder *encoder);

/* Free a ChatEncoder. */
void rsbpe_chat_encoder_free(RsBpeChatEncoder *encoder);

#ifdef __cplusplus
}
#endif

#endif /* RSBPE_H */
