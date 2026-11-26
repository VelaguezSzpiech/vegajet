// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// P-M4: the Jinja-subset parser recursed per nesting level in both the expression parser (nested
// parentheses / calls / literals) and the statement parser (nested {% if %} / {% for %}), so a
// crafted tokenizer.chat_template overflowed the stack. Both are depth-limited now.
#include "core/chat_template.hpp"
#include "tests/test_common.hpp"

#include <string>

using namespace vegajet;

namespace {

bool parse_ok(const std::string& src, std::string* err) {
    chat::Template t;
    return t.parse(src, err);
}

std::string nested_parens(size_t depth) {
    std::string s = "{{ ";
    s.append(depth, '(');
    s += "1";
    s.append(depth, ')');
    s += " }}";
    return s;
}

std::string nested_ifs(size_t depth) {
    std::string s;
    for (size_t i = 0; i < depth; ++i) s += "{% if true %}";
    s += "x";
    for (size_t i = 0; i < depth; ++i) s += "{% endif %}";
    return s;
}

} // namespace

int main() {
    {   // 200 000 '(' used to recurse until the stack was gone.
        std::string err;
        const bool ok = parse_ok(nested_parens(200000), &err);
        std::printf("  nested parens -> %s\n", err.c_str());
        VJ_CHECK(!ok);
        VJ_CHECK(err.find("nesting too deep") != std::string::npos);
    }
    {   // 100 000 nested {% if %} blocks likewise.
        std::string err;
        const bool ok = parse_ok(nested_ifs(100000), &err);
        std::printf("  nested if -> %s\n", err.c_str());
        VJ_CHECK(!ok);
        VJ_CHECK(err.find("nesting too deep") != std::string::npos);
    }
    {   // Realistic nesting still parses and renders.
        const std::string src = nested_ifs(5);
        std::string err, out;
        chat::Template t;
        VJ_CHECK(t.parse(src, &err));
        VJ_CHECK(t.render_json("{}", out, &err));
        VJ_CHECK(out == "x");
        const std::string expr = "{{ (" + std::string(8, '(') + "1 + 1" + std::string(8, ')') + ") * 2 }}";
        chat::Template e;
        VJ_CHECK(e.parse(expr, &err));
        VJ_CHECK(e.render_json("{}", out, &err));
        VJ_CHECK(out == "4");
    }

    std::printf("[PASS] test_parser_tpl_depth: expression and statement nesting depth-limited\n");
    return 0;
}
