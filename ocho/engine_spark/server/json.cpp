// json.cpp - minimal JSON parser/serializer implementation.
#include "json.h"
#include <cctype>
#include <cstdio>
#include <cstring>

namespace qf {

namespace {

struct P {
    const char *s;
    size_t n;
    size_t i = 0;
    std::string *err;

    bool fail(const char *msg) {
        if (err->empty()) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s at byte %zu", msg, i);
            *err = buf;
        }
        return false;
    }
    void ws() {
        while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    }
    bool expect(char c) {
        if (i >= n || s[i] != c) return fail("unexpected character");
        i++;
        return true;
    }

    bool parse_string(std::string &out) {
        if (!expect('"')) return false;
        out.clear();
        while (i < n) {
            unsigned char c = (unsigned char)s[i++];
            if (c == '"') return true;
            if (c == '\\') {
                if (i >= n) return fail("truncated escape");
                char e = s[i++];
                switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp == UINT32_MAX) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        // surrogate pair
                        if (i + 1 >= n || s[i] != '\\' || s[i + 1] != 'u')
                            return fail("lone high surrogate");
                        i += 2;
                        uint32_t lo = hex4();
                        if (lo == UINT32_MAX || lo < 0xDC00 || lo > 0xDFFF)
                            return fail("bad low surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return fail("lone low surrogate");
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: return fail("bad escape");
                }
            } else {
                if (c < 0x20) return fail("control char in string");
                out += (char)c;
            }
        }
        return fail("unterminated string");
    }

    uint32_t hex4() {
        if (i + 4 > n) { fail("truncated \\u escape"); return UINT32_MAX; }
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else { fail("bad hex digit"); return UINT32_MAX; }
        }
        return v;
    }

    static void append_utf8(std::string &out, uint32_t cp) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }

    bool parse_number(Json &out) {
        size_t start = i;
        if (i < n && s[i] == '-') i++;
        if (i >= n || !isdigit((unsigned char)s[i])) return fail("bad number");
        while (i < n && isdigit((unsigned char)s[i])) i++;
        if (i < n && s[i] == '.') {
            i++;
            if (i >= n || !isdigit((unsigned char)s[i])) return fail("bad number");
            while (i < n && isdigit((unsigned char)s[i])) i++;
        }
        if (i < n && (s[i] == 'e' || s[i] == 'E')) {
            i++;
            if (i < n && (s[i] == '+' || s[i] == '-')) i++;
            if (i >= n || !isdigit((unsigned char)s[i])) return fail("bad exponent");
            while (i < n && isdigit((unsigned char)s[i])) i++;
        }
        out = Json::make_num(strtod(std::string(s + start, i - start).c_str(), nullptr));
        return true;
    }

    bool parse_value(Json &out) {
        ws();
        if (i >= n) return fail("unexpected end of input");
        char c = s[i];
        if (c == '"') {
            std::string v;
            if (!parse_string(v)) return false;
            out = Json::make_str(std::move(v));
            return true;
        }
        if (c == '{') {
            i++;
            out = Json::make_obj();
            ws();
            if (i < n && s[i] == '}') { i++; return true; }
            while (true) {
                ws();
                std::string key;
                if (!parse_string(key)) return false;
                ws();
                if (!expect(':')) return false;
                Json val;
                if (!parse_value(val)) return false;
                out.obj.emplace_back(std::move(key), std::move(val));
                ws();
                if (i >= n) return fail("unterminated object");
                if (s[i] == ',') { i++; continue; }
                if (s[i] == '}') { i++; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            i++;
            out = Json::make_arr();
            ws();
            if (i < n && s[i] == ']') { i++; return true; }
            while (true) {
                Json val;
                if (!parse_value(val)) return false;
                out.arr.push_back(std::move(val));
                ws();
                if (i >= n) return fail("unterminated array");
                if (s[i] == ',') { i++; continue; }
                if (s[i] == ']') { i++; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == 't') {
            if (n - i >= 4 && memcmp(s + i, "true", 4) == 0) { i += 4; out = Json::make_bool(true); return true; }
            return fail("bad literal");
        }
        if (c == 'f') {
            if (n - i >= 5 && memcmp(s + i, "false", 5) == 0) { i += 5; out = Json::make_bool(false); return true; }
            return fail("bad literal");
        }
        if (c == 'n') {
            if (n - i >= 4 && memcmp(s + i, "null", 4) == 0) { i += 4; out = Json::make_null(); return true; }
            return fail("bad literal");
        }
        if (c == '-' || isdigit((unsigned char)c)) return parse_number(out);
        return fail("unexpected character");
    }
};

void dump_escaped(std::string &out, const std::string &s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += (char)c;
            }
        }
    }
    out += '"';
}

void dump_into(std::string &out, const Json &j) {
    switch (j.type) {
    case Json::NIL: out += "null"; break;
    case Json::BOOL: out += j.b ? "true" : "false"; break;
    case Json::NUM: {
        char buf[32];
        double v = j.num;
        if (v == (double)(long long)v && v >= -9e15 && v <= 9e15)
            snprintf(buf, sizeof(buf), "%lld", (long long)v);
        else
            snprintf(buf, sizeof(buf), "%.17g", v);
        out += buf;
        break;
    }
    case Json::STR: dump_escaped(out, j.str); break;
    case Json::ARR:
        out += '[';
        for (size_t k = 0; k < j.arr.size(); k++) {
            if (k) out += ',';
            dump_into(out, j.arr[k]);
        }
        out += ']';
        break;
    case Json::OBJ:
        out += '{';
        for (size_t k = 0; k < j.obj.size(); k++) {
            if (k) out += ',';
            dump_escaped(out, j.obj[k].first);
            out += ':';
            dump_into(out, j.obj[k].second);
        }
        out += '}';
        break;
    }
}

} // namespace

bool json_parse(const std::string &text, Json &out, std::string &err) {
    err.clear();
    P p{text.data(), text.size(), 0, &err};
    if (!p.parse_value(out)) return false;
    p.ws();
    if (p.i != p.n) return p.fail("trailing data");
    return true;
}

std::string json_dump(const Json &j) {
    std::string out;
    out.reserve(256);
    dump_into(out, j);
    return out;
}

} // namespace qf
