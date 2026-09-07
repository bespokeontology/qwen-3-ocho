// tokenizer_json_test.cpp - host-only unit test for the native
// tokenizer.json loader. Builds a tiny synthetic byte-level BPE
// tokenizer.json (same construction as GPT-2/Qwen2 bytes_to_unicode),
// writes it to a temp file, and checks encode/decode/special/eos behavior.
// No engine, no GPU, no model files.
#include "tokenizer_json.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

using qf::QfTokenizer;

static int g_fail = 0;
#define CHECK(cond) do { \
    if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } \
} while (0)

// GPT-2 bytes_to_unicode: printable bytes map to their own codepoint, the
// rest take 256+n. Returned as a UTF-8 string (the vocab key form).
static std::string b2u(unsigned b) {
    static std::string tab[256];
    static bool init = false;
    if (!init) {
        int n = 0;
        for (int i = 0; i < 256; i++) {
            bool printable = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174;
            unsigned cp = printable ? (unsigned)i : 256u + (unsigned)n++;
            std::string s;
            if (cp < 0x80) s += (char)cp;
            else { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
            tab[i] = s;
        }
        init = true;
    }
    return tab[b];
}

// Byte-unicode encode a raw byte string into vocab-key form.
static std::string key(const std::string &bytes) {
    std::string out;
    for (unsigned char c : bytes) out += b2u(c);
    return out;
}

int main() {
    // Vocab: single bytes we need, then merged tokens.
    // ids: h=0 e=1 l=2 o=3 ' '=4 w=5 r=6 d=7 0xC3=8 0xA9=9
    //      he=10 ll=11 hell=12 hello=13 Ġw=14 or=15 orl=16 orld=17 Ġworld=18
    //      <|im_start|>=900 <|im_end|>=901
    struct V { const char *bytes; int id; };
    std::string sp(1, ' '), c3(1, (char)0xC3), a9(1, (char)0xA9);
    std::vector<std::pair<std::string, int>> vocab = {
        {"h", 0}, {"e", 1}, {"l", 2}, {"o", 3}, {sp, 4}, {"w", 5}, {"r", 6}, {"d", 7},
        {c3, 8}, {a9, 9},
        {"he", 10}, {"ll", 11}, {"hell", 12}, {"hello", 13},
        {sp + "w", 14}, {"or", 15}, {"orl", 16}, {"orld", 17}, {sp + "world", 18},
    };
    std::vector<std::pair<std::string, std::string>> merges = {
        {"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o"},
        {sp, "w"}, {"o", "r"}, {"or", "l"}, {"orl", "d"}, {sp + "w", "orld"},
    };

    std::string js = "{ \"model\": { \"type\": \"BPE\", \"vocab\": {";
    for (size_t i = 0; i < vocab.size(); i++) {
        if (i) js += ",";
        js += "\"" + key(vocab[i].first) + "\": " + std::to_string(vocab[i].second);
    }
    js += "}, \"merges\": [";
    for (size_t i = 0; i < merges.size(); i++) {
        if (i) js += ",";
        js += "\"" + key(merges[i].first) + " " + key(merges[i].second) + "\"";
    }
    js += "] }, \"added_tokens\": ["
          "{\"id\": 900, \"content\": \"<|im_start|>\", \"special\": true},"
          "{\"id\": 901, \"content\": \"<|im_end|>\", \"special\": true}] }";

    char path[] = "/tmp/qf_tok_test_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { perror("mkstemp"); return 2; }
    FILE *f = fdopen(fd, "wb");
    fwrite(js.data(), 1, js.size(), f);
    fclose(f);

    auto tok = qf::qf_load_tokenizer_json(path);
    unlink(path);
    if (!tok) { fprintf(stderr, "FAIL: loader returned null\n"); return 1; }

    std::vector<int32_t> ids;

    CHECK(tok->encode("hello world", ids));
    CHECK(ids.size() == 2 && ids[0] == 13 && ids[1] == 18);

    // Special tokens are matched literally inside text.
    CHECK(tok->encode("<|im_start|>hello<|im_end|>", ids));
    CHECK(ids.size() == 3 && ids[0] == 900 && ids[1] == 13 && ids[2] == 901);

    // Chat-template-shaped input.
    CHECK(tok->encode("<|im_start|>user\nhello<|im_end|>\n", ids));
    CHECK(!ids.empty() && ids.front() == 900);
    CHECK(std::find(ids.begin(), ids.end(), 901) != ids.end());

    // Multi-byte UTF-8 ('é' = 0xC3 0xA9) encodes through single-byte tokens
    // and decodes back byte-exact.
    CHECK(tok->encode("\xC3\xA9", ids));
    CHECK(ids.size() == 2 && ids[0] == 8 && ids[1] == 9);
    std::string rt = tok->decode_token(ids[0]) + tok->decode_token(ids[1]);
    CHECK(rt == "\xC3\xA9");

    // Decode round trip of a full string.
    CHECK(tok->encode("hello world", ids));
    std::string whole;
    for (int32_t id : ids) whole += tok->decode_token(id);
    CHECK(whole == "hello world");

    CHECK(tok->eos() == 901);
    CHECK(tok->special("im_start") == 900);
    CHECK(tok->special("<|im_end|>") == 901);
    CHECK(tok->special("nope") == -1);

    if (g_fail) { fprintf(stderr, "%d check(s) failed\n", g_fail); return 1; }
    fprintf(stderr, "tokenizer_json_test: all checks passed\n");
    return 0;
}
