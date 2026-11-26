// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// P-M5: nested range() loops had no iteration budget, so a 90-byte template could keep the renderer
// busy for hours. A single render now stops at a bounded number of loop iterations.
#include "core/chat_template.hpp"
#include "tests/test_common.hpp"

#include <chrono>
#include <string>

using namespace vegajet;

int main() {
    {   // 2000 x 2000 = 4 000 000 iterations from a tiny template.
        chat::Template t;
        std::string err, out;
        const std::string src =
            "{% for a in range(2000) %}{% for b in range(2000) %}{% endfor %}{% endfor %}";
        VJ_CHECK(t.parse(src, &err));
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = t.render_json("{}", out, &err);
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  nested range(2000)^2 -> %s (%.2fs)\n", err.c_str(), secs);
        VJ_CHECK(!ok);
        VJ_CHECK(err.find("loop budget") != std::string::npos);
        VJ_CHECK(secs < 120.0);
    }
    {   // Normal nesting still renders, including loop metadata.
        chat::Template t;
        std::string err, out;
        VJ_CHECK(t.parse("{% for a in range(2) %}{% for b in range(3) %}{{ a }}{{ b }}"
                         "{{ loop.index }},{% endfor %}{% endfor %}", &err));
        VJ_CHECK(t.render_json("{}", out, &err));
        VJ_CHECK(out == "001,012,023,101,112,123,");
    }

    std::printf("[PASS] test_parser_tpl_loops: template loop iteration budget enforced\n");
    return 0;
}
