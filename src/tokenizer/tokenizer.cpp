#include "vllm/tokenizer/tokenizer.hpp"
#include "vllm/utils/logger.hpp"
#include "vllm/utils/json_utils.hpp"
#include "unicode_ranges.hpp"
#include "unicode_normalization.hpp"
#include <fstream>
#include <algorithm>
#include <array>
#include <sstream>
#include <cstdint>
#include <limits>

namespace vllm {

namespace {

// GPT-2/Qwen ByteLevel BPE represents each input byte as a Unicode code point.
// Convert between that representation and the original UTF-8 bytes.
std::array<uint32_t, 256> byte_to_unicode() {
    std::array<uint32_t, 256> table{};
    std::vector<bool> used(256, false);
    for (int b = 33; b <= 126; ++b) { table[b] = static_cast<uint32_t>(b); used[b] = true; }
    for (int b = 161; b <= 172; ++b) { table[b] = static_cast<uint32_t>(b); used[b] = true; }
    for (int b = 174; b <= 255; ++b) { table[b] = static_cast<uint32_t>(b); used[b] = true; }
    uint32_t extra = 256;
    for (int b = 0; b < 256; ++b) {
        if (!used[b]) table[b] = extra++;
    }
    return table;
}

std::array<int, 512> unicode_to_byte() {
    std::array<int, 512> table{};
    table.fill(-1);
    auto forward = byte_to_unicode();
    for (int b = 0; b < 256; ++b) {
        if (forward[b] < table.size()) table[forward[b]] = b;
    }
    return table;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::string byte_level_decode(const std::string& token) {
    auto table = unicode_to_byte();
    std::string out;
    for (size_t i = 0; i < token.size();) {
        unsigned char c = static_cast<unsigned char>(token[i]);
        uint32_t cp = 0;
        size_t n = 1;
        if (c < 0x80) cp = c;
        else if ((c & 0xE0) == 0xC0 && i + 1 < token.size()) {
            cp = (c & 0x1F) << 6 | (static_cast<unsigned char>(token[i + 1]) & 0x3F); n = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < token.size()) {
            cp = (c & 0x0F) << 12 | (static_cast<unsigned char>(token[i + 1]) & 0x3F) << 6 |
                 (static_cast<unsigned char>(token[i + 2]) & 0x3F); n = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < token.size()) {
            cp = (c & 0x07) << 18 | (static_cast<unsigned char>(token[i + 1]) & 0x3F) << 12 |
                 (static_cast<unsigned char>(token[i + 2]) & 0x3F) << 6 |
                 (static_cast<unsigned char>(token[i + 3]) & 0x3F); n = 4;
        } else {
            out.push_back(static_cast<char>(c)); ++i; continue;
        }
        if (cp < table.size() && table[cp] >= 0) out.push_back(static_cast<char>(table[cp]));
        else append_utf8(out, cp);
        i += n;
    }
    return out;
}

struct Codepoint {
    uint32_t value;
    size_t begin;
    size_t end;
};

std::vector<Codepoint> codepoints(const std::string& text) {
    std::vector<Codepoint> result;
    for (size_t i = 0; i < text.size();) {
        size_t start = i;
        unsigned char c = static_cast<unsigned char>(text[i++]);
        uint32_t value = c;
        int count = 0;
        if ((c & 0xe0) == 0xc0) { value = c & 0x1f; count = 1; }
        else if ((c & 0xf0) == 0xe0) { value = c & 0x0f; count = 2; }
        else if ((c & 0xf8) == 0xf0) { value = c & 0x07; count = 3; }
        if (i + count > text.size()) count = 0;
        for (int j = 0; j < count; ++j) {
            unsigned char next = static_cast<unsigned char>(text[i]);
            if ((next & 0xc0) != 0x80) break;
            value = (value << 6) | (next & 0x3f);
            ++i;
        }
        result.push_back({value, start, i});
    }
    return result;
}

uint8_t combining_class(uint32_t cp) {
    const auto* begin = std::begin(unicode::combining_classes);
    const auto* end = std::end(unicode::combining_classes);
    auto it = std::lower_bound(begin, end, cp,
                               [](const auto& entry, uint32_t value) { return entry.codepoint < value; });
    return it != end && it->codepoint == cp ? it->value : 0;
}

void decompose(uint32_t cp, std::vector<uint32_t>& out) {
    constexpr uint32_t s_base = 0xac00, l_base = 0x1100, v_base = 0x1161, t_base = 0x11a7;
    constexpr uint32_t l_count = 19, v_count = 21, t_count = 28;
    if (cp >= s_base && cp < s_base + l_count * v_count * t_count) {
        uint32_t s = cp - s_base;
        out.push_back(l_base + s / (v_count * t_count));
        out.push_back(v_base + (s / t_count) % v_count);
        if (s % t_count) out.push_back(t_base + s % t_count);
        return;
    }
    const auto* begin = std::begin(unicode::decompositions);
    const auto* end = std::end(unicode::decompositions);
    auto it = std::lower_bound(begin, end, cp,
                               [](const auto& entry, uint32_t value) { return entry.codepoint < value; });
    if (it == end || it->codepoint != cp) { out.push_back(cp); return; }
    for (size_t i = 0; i < it->length; ++i) out.push_back(unicode::decomposition_data[it->offset + i]);
}

uint32_t compose(uint32_t a, uint32_t b) {
    constexpr uint32_t s_base = 0xac00, l_base = 0x1100, v_base = 0x1161, t_base = 0x11a7;
    constexpr uint32_t l_count = 19, v_count = 21, t_count = 28;
    if (a >= l_base && a < l_base + l_count && b >= v_base && b < v_base + v_count)
        return s_base + (a - l_base) * v_count * t_count + (b - v_base) * t_count;
    if (a >= s_base && a < s_base + l_count * v_count * t_count &&
        (a - s_base) % t_count == 0 && b > t_base && b < t_base + t_count)
        return a + b - t_base;
    const auto* begin = std::begin(unicode::compositions);
    const auto* end = std::end(unicode::compositions);
    auto it = std::lower_bound(begin, end, std::pair<uint32_t, uint32_t>{a, b},
                               [](const auto& entry, const auto& key) {
                                   return std::pair<uint32_t, uint32_t>{entry.first, entry.second} < key;
                               });
    return it != end && it->first == a && it->second == b ? it->composed : 0;
}

std::string normalize_nfc(const std::string& text) {
    std::vector<uint32_t> decomposed;
    for (const auto& cp : codepoints(text)) decompose(cp.value, decomposed);
    for (size_t i = 1; i < decomposed.size(); ++i) {
        uint8_t cc = combining_class(decomposed[i]);
        size_t j = i;
        while (cc && j > 0 && combining_class(decomposed[j - 1]) > cc) {
            std::swap(decomposed[j], decomposed[j - 1]);
            --j;
        }
    }
    std::vector<uint32_t> composed;
    size_t starter = 0;
    uint8_t previous_cc = 0;
    for (uint32_t cp : decomposed) {
        uint8_t cc = combining_class(cp);
        uint32_t merged = composed.empty() ? 0 : compose(composed[starter], cp);
        if (merged && (previous_cc == 0 || previous_cc < cc)) {
            composed[starter] = merged;
        } else {
            if (cc == 0) starter = composed.size();
            composed.push_back(cp);
            previous_cc = cc;
        }
    }
    std::string result;
    for (uint32_t cp : composed) append_utf8(result, cp);
    return result;
}

// Follow the alternatives in Qwen's tokenizer.json Split regex, in their original order.
std::vector<std::string> pretokenize(const std::string& text) {
    const auto cps = codepoints(text);
    std::vector<std::string> pieces;
    for (size_t i = 0; i < cps.size();) {
        size_t end = i;
        uint32_t c = cps[i].value;
        if (c == '\'' && i + 1 < cps.size()) {
            static const std::vector<std::string> suffixes = {"re", "ve", "ll", "s", "t", "m", "d"};
            for (const auto& suffix : suffixes) {
                if (i + 1 + suffix.size() > cps.size()) continue;
                bool match = true;
                for (size_t j = 0; j < suffix.size(); ++j) {
                    uint32_t letter = cps[i + 1 + j].value;
                    if (letter >= 'A' && letter <= 'Z') letter += 'a' - 'A';
                    if (letter != static_cast<unsigned char>(suffix[j])) { match = false; break; }
                }
                if (match) { end = i + 1 + suffix.size(); break; }
            }
        }
        if (end == i) {
            size_t j = i;
            if (c != '\r' && c != '\n' && !unicode::is_letter(c) && !unicode::is_number(c) &&
                j + 1 < cps.size() && unicode::is_letter(cps[j + 1].value)) ++j;
            if (unicode::is_letter(cps[j].value)) {
                end = j + 1;
                while (end < cps.size() && unicode::is_letter(cps[end].value)) ++end;
            } else if (unicode::is_number(c)) {
                end = i + 1;
            } else {
                j = i;
                if (c == ' ' && j + 1 < cps.size() && !unicode::is_space(cps[j + 1].value) &&
                    !unicode::is_letter(cps[j + 1].value) && !unicode::is_number(cps[j + 1].value)) ++j;
                if (j < cps.size() && !unicode::is_space(cps[j].value) &&
                    !unicode::is_letter(cps[j].value) && !unicode::is_number(cps[j].value)) {
                    end = j + 1;
                    while (end < cps.size() && !unicode::is_space(cps[end].value) &&
                           !unicode::is_letter(cps[end].value) && !unicode::is_number(cps[end].value)) ++end;
                    while (end < cps.size() && (cps[end].value == '\r' || cps[end].value == '\n')) ++end;
                } else if (unicode::is_space(c)) {
                    j = i;
                    while (j < cps.size() && unicode::is_space(cps[j].value)) ++j;
                    size_t last_newline = j;
                    while (last_newline > i && cps[last_newline - 1].value != '\r' &&
                           cps[last_newline - 1].value != '\n') --last_newline;
                    if (last_newline > i) {
                        end = last_newline;
                    } else {
                        end = (j - i > 1 && j < cps.size()) ? j - 1 : j;
                    }
                }
            }
        }
        if (end == i) end = i + 1;
        pieces.push_back(text.substr(cps[i].begin, cps[end - 1].end - cps[i].begin));
        i = end;
    }
    return pieces;
}

}  // namespace

bool Tokenizer::is_stop_token(TokenId token) const {
    return token == eos_token();
}

SimpleTokenizer::SimpleTokenizer() {
    eos_token_ = -1;
}

bool SimpleTokenizer::load(const std::string& path) {
    id_to_token_.clear();
    token_to_id_.clear();
    merge_ranks_.clear();
    special_tokens_.clear();
    stop_tokens_.clear();
    std::string vocab_path;
    std::ifstream test;

    // Try tokenizer.json first
    vocab_path = path + "/tokenizer.json";
    test.open(vocab_path);
    if (test) {
        test.close();
        auto j = json::parse_file(vocab_path);
        auto& model = j["model"];
        auto& vocab_map = model["vocab"];
        for (auto& [token_str, token_id_val] : vocab_map.get_object()) {
            int id = token_id_val.get_int();
            if (static_cast<size_t>(id) >= id_to_token_.size()) {
                id_to_token_.resize(static_cast<size_t>(id) + 1);
            }
            id_to_token_[static_cast<size_t>(id)] = token_str;
            token_to_id_[token_str] = id;
        }
        if (model.contains("merges")) {
            auto& merges = model["merges"].get_array();
            for (size_t rank = 0; rank < merges.size(); ++rank) {
                std::string pair;
                if (merges[rank].is_string()) {
                    const auto& value = merges[rank].get_string();
                    size_t separator = value.find(' ');
                    if (separator == std::string::npos) continue;
                    pair = value.substr(0, separator) + '\0' + value.substr(separator + 1);
                } else {
                    const auto& value = merges[rank].get_array();
                    pair = value[0].get_string() + '\0' + value[1].get_string();
                }
                merge_ranks_.emplace(std::move(pair), static_cast<int32_t>(rank));
            }
        }
        // Special tokens
        if (j.contains("added_tokens")) {
            auto& added = j["added_tokens"];
            for (size_t i = 0; i < added.size(); i++) {
                auto& t = added.get_array()[i];
                int id = t["id"].get_int();
                const std::string& content = t["content"].get_string();
                if (static_cast<size_t>(id) >= id_to_token_.size()) {
                    id_to_token_.resize(static_cast<size_t>(id) + 1);
                }
                id_to_token_[static_cast<size_t>(id)] = content;
                token_to_id_[content] = id;
                if (t.contains("special") && t["special"].get_int() != 0)
                    special_tokens_.push_back(content);
            }
        }
        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto& a, const auto& b) { return a.size() > b.size(); });
        VLLM_INFO("Tokenizer: loaded {} tokens from tokenizer.json\n", id_to_token_.size());
    } else {
        // Fall back to vocab.json
        vocab_path = path + "/vocab.json";
        test.open(vocab_path);
        if (!test) {
            VLLM_ERROR("Tokenizer: neither tokenizer.json nor vocab.json found\n");
            return false;
        }
        test.close();
        auto j = json::parse_file(vocab_path);
        for (auto& [token_str, token_id_val] : j.get_object()) {
            int id = token_id_val.get_int();
            if (static_cast<size_t>(id) >= id_to_token_.size()) {
                id_to_token_.resize(static_cast<size_t>(id) + 1);
            }
            id_to_token_[static_cast<size_t>(id)] = token_str;
            token_to_id_[token_str] = id;
        }
        VLLM_INFO("Tokenizer: loaded {} tokens from vocab.json\n", id_to_token_.size());
    }

    // Set EOS token
    auto eos_it = token_to_id_.find("<|im_end|>");
    if (eos_it == token_to_id_.end()) eos_it = token_to_id_.find("<|endoftext|>");
    if (eos_it == token_to_id_.end()) eos_it = token_to_id_.find("</s>");
    if (eos_it == token_to_id_.end()) eos_it = token_to_id_.find("<eos>");
    if (eos_it != token_to_id_.end()) {
        eos_token_ = eos_it->second;
        stop_tokens_.push_back(eos_token_);
    } else eos_token_ = -1;
    auto endoftext = token_to_id_.find("<|endoftext|>");
    if (endoftext != token_to_id_.end() && endoftext->second != eos_token_)
        stop_tokens_.push_back(endoftext->second);

    VLLM_INFO("Tokenizer: vocab={}, eos={}\n", id_to_token_.size(), eos_token_);
    return true;
}

bool SimpleTokenizer::is_stop_token(TokenId token) const {
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
}

std::vector<TokenId> SimpleTokenizer::encode(const std::string& text) {
    std::vector<TokenId> tokens;
    auto encode_regular = [&](const std::string& segment) {
        for (const auto& piece : pretokenize(normalize_nfc(segment))) {
            std::vector<std::string> symbols;
            auto table = byte_to_unicode();
            for (unsigned char byte : piece) {
                symbols.emplace_back();
                append_utf8(symbols.back(), table[byte]);
            }
            while (symbols.size() > 1) {
                int32_t best_rank = std::numeric_limits<int32_t>::max();
                size_t best_at = 0;
                for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                    auto it = merge_ranks_.find(symbols[i] + '\0' + symbols[i + 1]);
                    if (it != merge_ranks_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_at = i;
                    }
                }
                if (best_rank == std::numeric_limits<int32_t>::max()) break;
                symbols[best_at] += symbols[best_at + 1];
                symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_at + 1));
            }
            for (const auto& symbol : symbols) {
                auto it = token_to_id_.find(symbol);
                if (it != token_to_id_.end()) tokens.push_back(it->second);
                else tokens.push_back(0);
            }
        }
    };
    size_t segment_begin = 0;
    for (size_t pos = 0; pos < text.size();) {
        const std::string* special = nullptr;
        for (const auto& candidate : special_tokens_) {
            if (text.compare(pos, candidate.size(), candidate) == 0) { special = &candidate; break; }
        }
        if (special) {
            encode_regular(text.substr(segment_begin, pos - segment_begin));
            tokens.push_back(token_to_id_.at(*special));
            pos += special->size();
            segment_begin = pos;
        } else ++pos;
    }
    encode_regular(text.substr(segment_begin));
    return tokens;
}

std::string SimpleTokenizer::decode(const std::vector<TokenId>& tokens) {
    std::ostringstream oss;
    for (auto t : tokens) oss << decode(t);
    return oss.str();
}

std::string SimpleTokenizer::decode(TokenId token) {
    if (token >= 0 && static_cast<size_t>(token) < id_to_token_.size()) {
        return byte_level_decode(id_to_token_[static_cast<size_t>(token)]);
    }
    return "[UNK]";
}

}  // namespace vllm
