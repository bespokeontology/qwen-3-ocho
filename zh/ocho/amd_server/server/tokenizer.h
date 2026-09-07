// tokenizer.h - tokenizer hook for the QF server.
//
// The engine core (qwenflash.h) works in token IDs only. Text<->token mapping
// is a server-side concern behind this interface:
//
//   - "ids" tokenizer: prompts arrive as pre-tokenized ID lists; decode is a
//     no-op placeholder. Useful for benchmarking and pre-tokenized clients.
//   - plugin tokenizer: dlopen'd shared library exporting the qf_tokenizer_v1
//     C ABI below (wrap HF tokenizers / sentencepiece / whatever you build).
//
// The server never bakes tokenizer internals into the engine; swap the .so
// (or the factory) without touching decode.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qf {

class QfTokenizer {
public:
    virtual ~QfTokenizer() = default;

    // Encode UTF-8 text into token IDs. Returns false on failure.
    virtual bool encode(const std::string &text, std::vector<int32_t> &out) = 0;

    // Decode a single token ID to its UTF-8 byte piece (may be a partial
    // UTF-8 sequence; callers buffer until a full codepoint is available).
    virtual std::string decode_token(int32_t id) = 0;

    // Look up a special token ID by name (e.g. "im_start"). -1 when unknown.
    virtual int32_t special(const char *name) const = 0;

    virtual int32_t eos() const = 0;
};

// ---- plugin ABI (versioned) -------------------------------------------------
// A tokenizer plugin exports:
//
//   extern "C" const qf_tokenizer_v1 *qf_tokenizer_v1();
//
// with the struct below. encode() writes up to `cap` IDs and returns the
// count, or -required_count when the buffer is too small, or 0 on error.
struct qf_tokenizer_v1 {
    void *(*create)(const char *model_dir);
    void (*destroy)(void *self);
    int (*encode)(void *self, const char *utf8, size_t len, int32_t *out, size_t cap);
    size_t (*decode_token)(void *self, int32_t id, char *out, size_t cap);
    int32_t (*special)(void *self, const char *name);
    int32_t (*eos)(void *self);
};

// spec:
//   "ids"            -> pre-tokenized ID lists ("1,2,3" or "[1, 2, 3]")
//   "json"           -> native Qwen byte-level BPE from model_dir/tokenizer.json
//   "/path/tokenizer.json" -> native loader on an explicit file
//   "auto"           -> model_dir/tokenizer.json if present, else
//                       $QF_TOKENIZER_LIB, else "ids"
//   "/path/lib.so"   -> plugin tokenizer via dlopen
// model_dir is passed to the plugin create() (for tokenizer.json lookup).
// Returns nullptr on failure (message printed to stderr).
std::unique_ptr<QfTokenizer> qf_load_tokenizer(const std::string &spec, const std::string &model_dir);

} // namespace qf
