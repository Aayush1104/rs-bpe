#ifndef RSBPE_H
#define RSBPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque tokenizer handle. */
typedef struct RsBpeTokenizer RsBpeTokenizer;

/* Result of an encode operation. */
typedef struct {
    uint32_t *tokens;   /* Rust-allocated token array; free with rsbpe_free_tokens */
    size_t    len;      /* Number of tokens */
    int32_t   error_code; /* 0 = success, 1 = invalid UTF-8 */
} RsBpeEncodeResult;

/* Create a DeepSeek tokenizer. Caller owns the returned pointer. */
RsBpeTokenizer *rsbpe_new_deepseek(void);

/* Encode UTF-8 text into token IDs. text_ptr need not be null-terminated. */
RsBpeEncodeResult rsbpe_encode(const RsBpeTokenizer *handle,
                               const char *text_ptr,
                               size_t text_len);

/* Free a token array returned by rsbpe_encode. */
void rsbpe_free_tokens(uint32_t *tokens, size_t len);

/* Free a tokenizer handle returned by rsbpe_new_deepseek. */
void rsbpe_free_tokenizer(RsBpeTokenizer *handle);

#ifdef __cplusplus
}
#endif

#endif /* RSBPE_H */
