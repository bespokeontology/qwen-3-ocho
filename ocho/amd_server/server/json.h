// json.h - minimal JSON parser/serializer for the QF server.
// No external dependencies; strict enough for API payloads.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qf {

struct Json;
using JsonPtr = std::shared_ptr<Json>;

struct Json {
    enum Type { NIL, BOOL, NUM, STR, ARR, OBJ } type = NIL;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    static Json make_null() { return Json{}; }
    static Json make_bool(bool v) { Json j; j.type = BOOL; j.b = v; return j; }
    static Json make_num(double v) { Json j; j.type = NUM; j.num = v; return j; }
    static Json make_str(std::string v) { Json j; j.type = STR; j.str = std::move(v); return j; }
    static Json make_arr() { Json j; j.type = ARR; return j; }
    static Json make_obj() { Json j; j.type = OBJ; return j; }

    bool is_null() const { return type == NIL; }
    bool is_obj() const { return type == OBJ; }
    bool is_arr() const { return type == ARR; }
    bool is_str() const { return type == STR; }
    bool is_num() const { return type == NUM; }
    bool is_bool() const { return type == BOOL; }

    // Object access. Returns nullptr when absent or not an object.
    const Json *get(const char *key) const {
        if (type != OBJ) return nullptr;
        for (auto &kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    bool has(const char *key) const { return get(key) != nullptr; }

    // Typed getters with defaults (type mismatch yields the default).
    std::string get_str(const char *key, const std::string &dflt = "") const {
        const Json *v = get(key);
        return (v && v->type == STR) ? v->str : dflt;
    }
    double get_num(const char *key, double dflt = 0) const {
        const Json *v = get(key);
        return (v && v->type == NUM) ? v->num : dflt;
    }
    long get_int(const char *key, long dflt = 0) const {
        const Json *v = get(key);
        return (v && v->type == NUM) ? (long)v->num : dflt;
    }
    bool get_bool(const char *key, bool dflt = false) const {
        const Json *v = get(key);
        if (!v) return dflt;
        if (v->type == BOOL) return v->b;
        if (v->type == NUM) return v->num != 0;
        return dflt;
    }
    void set(const std::string &key, Json v) {
        if (type != OBJ) { type = OBJ; obj.clear(); }
        for (auto &kv : obj)
            if (kv.first == key) { kv.second = std::move(v); return; }
        obj.emplace_back(key, std::move(v));
    }
    void push(Json v) {
        if (type != ARR) { type = ARR; arr.clear(); }
        arr.push_back(std::move(v));
    }
};

// Parse UTF-8 JSON. On error returns false and fills err with a message.
bool json_parse(const std::string &text, Json &out, std::string &err);

// Serialize compact JSON (strings escaped, UTF-8 preserved).
std::string json_dump(const Json &j);

} // namespace qf
