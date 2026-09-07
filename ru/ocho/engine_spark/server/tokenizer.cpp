// tokenizer.cpp - tokenizer hook implementations.
#include "tokenizer.h"
#include "tokenizer_json.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <sys/stat.h>

namespace qf {

namespace {

constexpr int32_t kDefaultEos = 248044;  // Qwen3.8-Flash-Next EOS per SEMANTICS.md

static bool file_exists(const std::string &path) {
    struct stat sb;
    return stat(path.c_str(), &sb) == 0;
}

// Pre-tokenized prompts: "1,2,3" or "[1, 2, 3]". decode_token returns the raw
// id as text so streams stay byte-exact debuggable without a vocab file.
class IdListTokenizer : public QfTokenizer {
public:
    bool encode(const std::string &text, std::vector<int32_t> &out) override {
        out.clear();
        const char *p = text.c_str();
        while (*p) {
            while (*p && (isspace((unsigned char)*p) || *p == ',' || *p == '[' || *p == ']')) p++;
            if (!*p) break;
            char *end = nullptr;
            long v = strtol(p, &end, 10);
            if (end == p) return false;
            if (v < 0 || v > INT32_MAX) return false;
            out.push_back((int32_t)v);
            p = end;
        }
        return !out.empty();
    }
    std::string decode_token(int32_t id) override {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", id);
        return buf;
    }
    int32_t special(const char *) const override { return -1; }
    int32_t eos() const override { return kDefaultEos; }
};

class PluginTokenizer : public QfTokenizer {
public:
    PluginTokenizer(void *lib, const qf_tokenizer_v1 *api, void *inst)
        : lib_(lib), api_(api), inst_(inst) {}
    ~PluginTokenizer() override {
        if (api_ && inst_) api_->destroy(inst_);
        if (lib_) dlclose(lib_);
    }

    bool encode(const std::string &text, std::vector<int32_t> &out) override {
        size_t cap = text.size() + 16;  // tokens never exceed bytes for BPE
        out.resize(cap);
        int n = api_->encode(inst_, text.data(), text.size(), out.data(), cap);
        if (n < 0) {
            cap = (size_t)(-n);
            out.resize(cap);
            n = api_->encode(inst_, text.data(), text.size(), out.data(), cap);
            if (n < 0) return false;
        }
        if (n == 0) return false;
        out.resize((size_t)n);
        return true;
    }
    std::string decode_token(int32_t id) override {
        char buf[4096];
        size_t n = api_->decode_token(inst_, id, buf, sizeof(buf));
        if (n <= sizeof(buf)) return std::string(buf, n);
        std::string big(n, '\0');
        api_->decode_token(inst_, id, big.data(), big.size());
        return big;
    }
    int32_t special(const char *name) const override { return api_->special(inst_, name); }
    int32_t eos() const override {
        int32_t e = api_->eos(inst_);
        return e >= 0 ? e : kDefaultEos;
    }

private:
    void *lib_;
    const qf_tokenizer_v1 *api_;
    void *inst_;
};

} // namespace

std::unique_ptr<QfTokenizer> qf_load_tokenizer(const std::string &spec_in, const std::string &model_dir) {
    std::string spec = spec_in;
    if (spec.empty() || spec == "auto") {
        // Prefer the concrete tokenizer.json shipped with the checkpoint;
        // fall back to a plugin library, then to pre-tokenized id lists.
        if (file_exists(model_dir + "/tokenizer.json")) {
            spec = "json";
        } else {
            const char *env = getenv("QF_TOKENIZER_LIB");
            spec = (env && *env) ? env : "ids";
        }
    }
    if (spec == "ids") {
        fprintf(stderr, "tokenizer: using id-list mode (prompts must be token IDs)\n");
        return std::make_unique<IdListTokenizer>();
    }
    std::string json_path;
    if (spec == "json") json_path = model_dir + "/tokenizer.json";
    else if (spec.size() >= 5 && spec.compare(spec.size() - 5, 5, ".json") == 0) json_path = spec;
    if (!json_path.empty())
        return qf_load_tokenizer_json(json_path);
    void *lib = dlopen(spec.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "tokenizer: dlopen(%s): %s\n", spec.c_str(), dlerror());
        return nullptr;
    }
    auto get = (const qf_tokenizer_v1 *(*)())dlsym(lib, "qf_tokenizer_v1");
    if (!get) {
        fprintf(stderr, "tokenizer: %s lacks qf_tokenizer_v1()\n", spec.c_str());
        dlclose(lib);
        return nullptr;
    }
    const qf_tokenizer_v1 *api = get();
    if (!api || !api->create || !api->destroy || !api->encode || !api->decode_token) {
        fprintf(stderr, "tokenizer: %s has an incomplete v1 ABI\n", spec.c_str());
        dlclose(lib);
        return nullptr;
    }
    void *inst = api->create(model_dir.c_str());
    if (!inst) {
        fprintf(stderr, "tokenizer: plugin create() failed for %s\n", model_dir.c_str());
        dlclose(lib);
        return nullptr;
    }
    fprintf(stderr, "tokenizer: loaded plugin %s\n", spec.c_str());
    return std::unique_ptr<QfTokenizer>(new PluginTokenizer(lib, api, inst));
}

} // namespace qf
