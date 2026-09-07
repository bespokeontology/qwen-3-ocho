// tokenizer_json.h - native HuggingFace tokenizer.json loader for Qwen.
//
// Concrete tokenizer for the server: loads <model_dir>/tokenizer.json (HF
// "fast" serialization) and implements Qwen2-style byte-level BPE directly,
// with no external tokenizer dependency:
//
//   - model.vocab:  token string -> id (GPT-2 byte-unicode encoded keys)
//   - model.merges: "A B" pair list (or ["A","B"] arrays) giving merge ranks
//   - added_tokens: <|im_start|> / <|im_end|> / <|endoftext|> and friends,
//     matched greedily in the input before BPE (the chat template emits them
//     literally, so the tokenizer must recognize them)
//
// Pre-tokenization follows the GPT-2/Qwen2 regex
//   '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// implemented as a hand-rolled scanner: case-sensitive contraction suffixes,
// runs of letters / digits / other, and the trailing-whitespace rule. The
// \p{L} class is approximated at byte level: every byte >= 0x80 counts as a
// letter, which is exact for pure-ASCII and CJK-only text and differs from
// the reference only on mixed-letter/punctuation non-ASCII runs (e.g. a CJK
// run directly followed by a CJK full-stop merges into one pre-token; BPE
// still produces correct tokens inside such merged runs whenever the merge
// table covers them, but segmentation can differ from HF tokenizers there).
//
// decode_token returns the raw byte piece for a token id; multi-byte UTF-8
// characters can be split across tokens, so callers buffer bytes until a
// full codepoint is available (the scheduler does this for SSE streaming).
#pragma once
#include "tokenizer.h"

namespace qf {

// Load a HF tokenizer.json from `path` (usually <model_dir>/tokenizer.json).
// Returns nullptr on failure (message printed to stderr).
std::unique_ptr<QfTokenizer> qf_load_tokenizer_json(const std::string &path);

} // namespace qf
