// tokenizer_json.cpp - native HuggingFace tokenizer.json loader (Qwen
// byte-level BPE). See tokenizer_json.h for the format notes and the
// pre-tokenizer approximation.
#include "tokenizer_json.h"
#include "json.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qf {
namespace {

constexpr int32_t kDefaultEos = 248044;  // Qwen3.8-Flash-Next EOS per SEMANTICS.md

// ---- GPT-2 bytes<->unicode alphabet -----------------------------------------
// Printable bytes map to their own codepoint; the remaining 68 bytes take
// codepoints 256..323 in increasing byte order (OpenAI bytes_to_unicode()).
struct ByteAlphabet {
    int cp2byte[324];  // codepoint -> byte, -1 when not in the alphabet
    ByteAlphabet() {
        std::fill(cp2byte, cp2byte + 324, -1);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
            int cp = printable ? b : 256 + n++;
            cp2byte[cp] = b;
        }
    }
};
static const ByteAlphabet g_alpha;

// Decode one UTF-8 codepoint at *i (advanced past it). Returns -1 on
// malformed input (caller decides; we treat the raw byte as its own
// codepoint so encoding never fails on odd bytes).
static uint32_t next_cp(const std::string &s, size_t *i) {
    unsigned char c = (unsigned char)s[*i];
    if (c < 0x80) { (*i)++; return c; }
    int len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : 2;
    if (*i + (size_t)len > s.size()) { (*i)++; return c; }
    uint32_t cp = c & ((1u << (7 - len)) - 1);
    for (int k = 1; k < len; k++) {
        unsigned char t = (unsigned char)s[*i + k];
        if ((t & 0xC0) != 0x80) { (*i)++; return c; }
        cp = (cp << 6) | (t & 0x3F);
    }
    *i += (size_t)len;
    return cp;
}

// Convert a byte-unicode vocab string (as stored in tokenizer.json) to the
// raw byte sequence it encodes. Codepoints outside the alphabet (added
// tokens with plain UTF-8 content) are copied through as raw UTF-8 bytes.
static std::string tokstr_to_bytes(const std::string &utf8) {
    std::string out;
    out.reserve(utf8.size());
    for (size_t i = 0; i < utf8.size();) {
        size_t at = i;
        uint32_t cp = next_cp(utf8, &i);
        if (cp < 324 && g_alpha.cp2byte[cp] >= 0)
            out += (char)g_alpha.cp2byte[cp];
        else
            out.append(utf8, at, i - at);  // pass through raw UTF-8
    }
    return out;
}

// ---- pre-tokenizer character classes ----------------------------------------
static bool is_ws(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
static bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
// Bytes >= 0x80 approximate \p{L} (see header note).
static bool is_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}

// Case-sensitive contraction suffixes from the GPT-2 pattern
// '(?:[sdmt]|ll|ve|re). Returns the suffix length (2 or 3 incl. apostrophe)
// or 0.
static int contraction_len(const std::string &s, size_t i) {
    if (s[i] != '\'' || i + 1 >= s.size()) return 0;
    char a = s[i + 1];
    if (i + 2 < s.size()) {
        char b = s[i + 2];
        if ((a == 'l' && b == 'l') || (a == 'v' && b == 'e') || (a == 'r' && b == 'e'))
            return 3;
    }
    if (a == 's' || a == 't' || a == 'd' || a == 'm') return 2;
    return 0;
}

class JsonTokenizer : public QfTokenizer {
public:
    bool load(const std::string &path) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) { fprintf(stderr, "tokenizer.json: cannot open %s\n", path.c_str()); return false; }
        fseek(f, 0, SEEK_END);
        long n = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::string text((size_t)n, '\0');
        size_t rd = fread(&text[0], 1, (size_t)n, f);
        fclose(f);
        text.resize(rd);

        Json root;
        std::string err;
        if (!json_parse(text, root, err) || !root.is_obj()) {
            fprintf(stderr, "tokenizer.json: parse failed: %s\n", err.c_str());
            return false;
        }
        const Json *model = root.get("model");
        if (!model || !model->is_obj()) {
            fprintf(stderr, "tokenizer.json: no model object\n");
            return false;
        }
        std::string mtype = model->get_str("type");
        if (!mtype.empty() && mtype != "BPE") {
            fprintf(stderr, "tokenizer.json: unsupported model type %s\n", mtype.c_str());
            return false;
        }
        const Json *vocab = model->get("vocab");
        if (!vocab || !vocab->is_obj()) {
            fprintf(stderr, "tokenizer.json: no model.vocab\n");
            return false;
        }
        int32_t max_id = -1;
        for (const auto &kv : vocab->obj) {
            if (kv.second.type != Json::NUM) continue;
            int32_t id = (int32_t)kv.second.num;
            std::string bytes = tokstr_to_bytes(kv.first);
            tok2id_[bytes] = id;
            if (id > max_id) max_id = id;
        }
        if (max_id < 0) return false;
        id2tok_.assign((size_t)max_id + 1, std::string());
        for (const auto &kv : tok2id_) id2tok_[(size_t)kv.second] = kv.first;

        // Single-byte fallback ids for the BPE seed pass.
        int missing = 0;
        for (int b = 0; b < 256; b++) {
            auto it = tok2id_.find(std::string(1, (char)b));
            if (it != tok2id_.end()) byte_id_[b] = it->second;
            else { byte_id_[b] = -1; missing++; }
        }
        if (missing)
            fprintf(stderr, "tokenizer.json: warning: %d/256 byte tokens missing from vocab\n", missing);

        const Json *merges = model->get("merges");
        if (merges && merges->is_arr()) {
            int32_t rank = 0;
            for (const auto &m : merges->arr) {
                std::string a, b;
                if (m.is_str()) {
                    size_t sp = m.str.find(' ');
                    if (sp == std::string::npos) { rank++; continue; }
                    a = m.str.substr(0, sp);
                    b = m.str.substr(sp + 1);
                } else if (m.is_arr() && m.arr.size() == 2) {
                    a = m.arr[0].str;
                    b = m.arr[1].str;
                } else { rank++; continue; }
                std::string ab = tokstr_to_bytes(a), bb = tokstr_to_bytes(b);
                auto ia = tok2id_.find(ab), ib = tok2id_.find(bb);
                auto im = tok2id_.find(ab + bb);
                if (ia != tok2id_.end() && ib != tok2id_.end() && im != tok2id_.end()) {
                    uint64_t key = ((uint64_t)(uint32_t)ia->second << 32) | (uint32_t)ib->second;
                    merges_[key] = {rank, im->second};
                }
                rank++;
            }
        }

        const Json *added = root.get("added_tokens");
        if (added && added->is_arr()) {
            for (const auto &t : added->arr) {
                if (!t.is_obj()) continue;
                std::string content = t.get_str("content");
                long id = t.get_int("id", -1);
                if (content.empty() || id < 0) continue;
                specials_.emplace_back(content, (int32_t)id);
                if ((size_t)id < id2tok_.size()) id2tok_[(size_t)id] = content;
            }
            // Longest match wins when specials overlap.
            std::sort(specials_.begin(), specials_.end(),
                      [](const auto &x, const auto &y) { return x.first.size() > y.first.size(); });
        }

        int32_t e = special("im_end");
        if (e < 0) e = special("endoftext");
        eos_ = e >= 0 ? e : kDefaultEos;

        fprintf(stderr, "tokenizer: loaded %s (%zu vocab, %zu merges, %zu added tokens, eos %d)\n",
                path.c_str(), tok2id_.size(), merges_.size(), specials_.size(), eos_);
        return true;
    }

    bool encode(const std::string &text, std::vector<int32_t> &out) override {
        out.clear();
        size_t i = 0, n = text.size();
        while (i < n) {
            // Earliest added-token occurrence at/after i; specials_ is sorted
            // longest-first so a strict < prefers the longest on ties.
            size_t best_pos = std::string::npos;
            int best = -1;
            for (size_t s = 0; s < specials_.size(); s++) {
                size_t p = text.find(specials_[s].first, i);
                if (p < best_pos) { best_pos = p; best = (int)s; }
            }
            size_t seg_end = (best == -1) ? n : best_pos;
            encode_segment(text, i, seg_end, out);
            if (best == -1) break;
            out.push_back(specials_[(size_t)best].second);
            i = best_pos + specials_[(size_t)best].first.size();
        }
        return true;
    }

    std::string decode_token(int32_t id) override {
        if (id >= 0 && (size_t)id < id2tok_.size()) return id2tok_[(size_t)id];
        return std::string();
    }

    int32_t special(const char *name) const override {
        std::string want = name;
        std::string wrapped = "<|" + want + "|>";
        for (const auto &s : specials_)
            if (s.first == want || s.first == wrapped) return s.second;
        return -1;
    }

    int32_t eos() const override { return eos_; }

private:
    struct Merge { int32_t rank; int32_t out; };
    std::unordered_map<std::string, int32_t> tok2id_;
    std::vector<std::string> id2tok_;
    std::unordered_map<uint64_t, Merge> merges_;
    std::vector<std::pair<std::string, int32_t>> specials_;
    int32_t byte_id_[256] = {};
    int32_t eos_ = kDefaultEos;

    // GPT-2/Qwen2 pre-tokenizer over [begin, end); BPE each chunk into out.
    void encode_segment(const std::string &text, size_t begin, size_t end,
                        std::vector<int32_t> &out) {
        size_t i = begin;
        while (i < end) {
            unsigned char c = (unsigned char)text[i];
            if (c == '\'') {
                int cl = contraction_len(text, i);
                if (cl && i + (size_t)cl <= end) { bpe(text, i, i + (size_t)cl, out); i += (size_t)cl; continue; }
            }
            if (is_ws(c)) {
                size_t j = i;
                while (j < end && is_ws((unsigned char)text[j])) j++;
                if (j == end) { bpe(text, i, j, out); i = j; continue; }  // \s+(?!\S): trailing run
                if (j - i > 1) { bpe(text, i, j - 1, out); i = j - 1; continue; }  // last space attaches
                // single whitespace before text: handled by the class runs below
            }
            size_t start = i, k = i;
            if (is_ws((unsigned char)text[k])) k++;  // optional single-space prefix
            if (k >= end) { bpe(text, start, k, out); i = k; continue; }
            unsigned char fc = (unsigned char)text[k];
            if (is_letter(fc))      while (k < end && is_letter((unsigned char)text[k])) k++;
            else if (is_digit(fc))  while (k < end && is_digit((unsigned char)text[k])) k++;
            else                    while (k < end && !is_ws((unsigned char)text[k]) &&
                                           !is_letter((unsigned char)text[k]) &&
                                           !is_digit((unsigned char)text[k])) k++;
            bpe(text, start, k, out);
            i = k;
        }
    }

    // Byte-level BPE over one pre-token [begin, end): seed from single-byte
    // tokens, then repeatedly merge the lowest-rank adjacent pair.
    void bpe(const std::string &text, size_t begin, size_t end, std::vector<int32_t> &out) {
        size_t len = end - begin;
        std::vector<int32_t> ids;
        ids.reserve(len);
        for (size_t k = begin; k < end; k++) {
            int32_t id = byte_id_[(unsigned char)text[k]];
            if (id < 0) {
                // Vocab lacks this byte (broken tokenizer.json): keep going
                // via the whole-chunk lookup below rather than dropping data.
                auto it = tok2id_.find(text.substr(begin, len));
                if (it != tok2id_.end()) out.push_back(it->second);
                return;
            }
            ids.push_back(id);
        }
        while (ids.size() > 1) {
            int32_t best_rank = INT32_MAX;
            int32_t best_out = -1;
            uint32_t best_a = 0, best_b = 0;
            for (size_t k = 0; k + 1 < ids.size(); k++) {
                uint64_t key = ((uint64_t)(uint32_t)ids[k] << 32) | (uint32_t)ids[k + 1];
                auto it = merges_.find(key);
                if (it != merges_.end() && it->second.rank < best_rank) {
                    best_rank = it->second.rank;
                    best_out = it->second.out;
                    best_a = (uint32_t)ids[k];
                    best_b = (uint32_t)ids[k + 1];
                }
            }
            if (best_out < 0) break;
            std::vector<int32_t> next;
            next.reserve(ids.size());
            for (size_t k = 0; k < ids.size(); k++) {
                if (k + 1 < ids.size() && (uint32_t)ids[k] == best_a && (uint32_t)ids[k + 1] == best_b) {
                    next.push_back(best_out);
                    k++;
                } else {
                    next.push_back(ids[k]);
                }
            }
            ids.swap(next);
        }
        for (int32_t id : ids) out.push_back(id);
    }
};

} // namespace

std::unique_ptr<QfTokenizer> qf_load_tokenizer_json(const std::string &path) {
    auto t = std::unique_ptr<JsonTokenizer>(new JsonTokenizer());
    if (!t->load(path)) return nullptr;
    return std::unique_ptr<QfTokenizer>(std::move(t));
}

} // namespace qf
