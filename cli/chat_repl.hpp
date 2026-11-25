// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// `vegajet chat`: an interactive REPL against any OpenAI-compatible server (a running `vegajet serve`, the swarm
// router, ...). Replies stream as they are generated (reasoning dimmed on a terminal).
//
//   vegajet chat [--url http://127.0.0.1:8080] [--model NAME] [--system TEXT] [--max-tokens N] [--temperature T]
//                [--no-stream]
//
// Commands: /stats (turns, prompt / completion tokens, last TTFT and tok/s), /clear (forget the conversation, keep
// the system prompt), /save <file> and /load <file> (the conversation as JSON), /system <text>, /help, /quit.
#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace vegajet::cli {

struct ChatConfig {
    std::string url = "http://127.0.0.1:8080";
    std::string model;          // "" = server default
    std::string system;
    int32_t max_tokens = 512;
    double temperature = -1.0;  // < 0: server default
    bool stream = true;
    bool color = false;
};

struct ChatTurnText {
    std::string role, content;
};

class ChatSession {
public:
    explicit ChatSession(ChatConfig cfg);
    // One input line (a command or a user message). False when the session should end (/quit, /exit).
    bool handle(const std::string& line, std::ostream& out);

    struct Stats {
        uint32_t turns = 0, errors = 0;
        uint64_t prompt_tokens = 0, completion_tokens = 0;
        double last_ttft_ms = 0.0, last_tok_s = 0.0, total_gen_s = 0.0;
    };
    const Stats& stats() const { return st_; }
    const std::vector<ChatTurnText>& history() const { return hist_; }
    const ChatConfig& config() const { return cfg_; }
    bool save(const std::string& path, std::string* err) const;
    bool load(const std::string& path, std::string* err);
    std::string request_json(const std::string& user) const;  // the request body for the next turn

private:
    bool send(const std::string& user, std::ostream& out);
    ChatConfig cfg_;
    std::vector<ChatTurnText> hist_;
    Stats st_;
};

int run_chat(const std::vector<std::string>& args);

}  // namespace vegajet::cli
