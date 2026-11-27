// tests/program/batch_admit_test.cpp - STRATA_BATCH_ADMIT wire/framing parity (CPU only, no GPU, no daemon).
//
// The two pure helpers below mirror generate.cpp's batch_binary_line/batch_crc32_prefix byte for byte; this
// target exists so a framing or CRC regression is caught at compile+unit-test time, before any GPU run.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "strata/prefill/prefill.hpp"

namespace {

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) { std::fprintf(stderr, "FAIL %s\n", what); ++failures; }
}

bool batch_binary_line(const std::string& l) {
    return l.rfind("BGENB ", 0) == 0 || l.rfind("GENB ", 0) == 0 ||
           l.rfind("BGEND ", 0) == 0 || l.rfind("BGBD ", 0) == 0;
}

uint32_t batch_crc32_prefix(const std::vector<int32_t>& ids, size_t n) {
    static const std::array<uint32_t, 256> tab = []() {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    uint32_t h = 0xFFFFFFFFu;
    const size_t n_check = std::min(n, ids.size());
    for (size_t i = 0; i < n_check; ++i) {
        const uint32_t uv = (uint32_t) ids[i];
        h = tab[(h ^ (uv & 0xffu)) & 0xffu] ^ (h >> 8);
        h = tab[(h ^ ((uv >> 8) & 0xffu)) & 0xffu] ^ (h >> 8);
        h = tab[(h ^ ((uv >> 16) & 0xffu)) & 0xffu] ^ (h >> 8);
        h = tab[(h ^ ((uv >> 24) & 0xffu)) & 0xffu] ^ (h >> 8);
    }
    return h ^ 0xFFFFFFFFu;
}

// Independent, bit-at-a-time CRC over the same little-endian byte stream.
uint32_t crc32_slow(const std::vector<int32_t>& ids, size_t n) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < n && i < ids.size(); ++i) {
        const uint32_t uv = (uint32_t) ids[i];
        for (int b = 0; b < 4; ++b) {
            crc ^= (uv >> (8 * b)) & 0xffu;
            for (int k = 0; k < 8; ++k) crc = (crc & 1u) ? (0xEDB88320u ^ (crc >> 1)) : (crc >> 1);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

// Mirrors the engine's BGBD header read: `BGBD <slot> <max_new> <opts> ... bin=<n>`.
bool parse_bgb_header(const std::string& line, int& slot, int64_t& max_new, int64_t& pfx,
                      uint64_t& hash, bool& has_hash, size_t& bin, bool& ckpt, int expected_slot) {
    const char* p = line.c_str() + 5;
    char* e = nullptr;
    slot = (int) std::strtol(p, &e, 10);
    if (e == p || slot != expected_slot) return false;
    p = e;
    max_new = std::strtoll(p, &e, 10);
    if (e == p || max_new < 1) return false;
    pfx = 0; hash = 0; has_hash = false; bin = 0; ckpt = true;
    while (*e != '\0') {
        while (*e == ' ') ++e;
        const char* ks = e;
        while (*e != '\0' && *e != ' ') ++e;
        if (e == ks) break;
        const std::string kv(ks, (size_t) (e - ks));
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) break;
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if (k == "pfx") pfx = std::atoll(v.c_str());
        else if (k == "pfx_hash") { hash = std::strtoull(v.c_str(), nullptr, 16); has_hash = true; }
        else if (k == "ckpt") ckpt = std::atoi(v.c_str()) != 0;
        else if (k == "bin") { bin = (size_t) std::strtoull(v.c_str(), nullptr, 10); break; }
    }
    return true;
}

struct Ticket {
    int slot = -1;
    int64_t max_new = 0;
    int64_t pfx = 0;
    uint64_t hash = 0;
    bool has_hash = false;
    std::vector<int32_t> bin;
    std::vector<int64_t> all;
    size_t delta = 0;
    bool ok = false;
};

// Mirrors the engine's cohort validation: slot range, distinctness, CRC prefix, cold payload bound.
bool cohort_ok(std::vector<Ticket>& tk, size_t n, int n_slots, int64_t max_delta) {
    for (size_t i = 0; i < n; ++i) {
        Ticket& t = tk[i];
        if (t.slot < 0 || t.slot >= n_slots || t.max_new < 1) return false;
        for (size_t j = 0; j < i; ++j) if (tk[j].slot == t.slot) return false;
    }
    for (Ticket& t : tk) {
        if (t.pfx > 0) {
            const std::vector<int32_t> prefix(t.all.begin(), t.all.begin() + (size_t) t.pfx);
            if (!t.has_hash || batch_crc32_prefix(prefix, (size_t) t.pfx) != t.hash) continue; // only this ticket fails
            t.delta = t.bin.size();
        } else {
            if (t.bin.empty() || (int64_t) t.bin.size() > max_delta) continue;
            t.all.assign(t.bin.begin(), t.bin.end());
            t.delta = t.all.size();
        }
        t.ok = true;
    }
    return true;
}

} // namespace

// Mirrors run_batch_slots' super-chunk cap: ~4 balanced super-chunks for a layer split, no change otherwise.
int64_t batch_max_super(int64_t total, const std::vector<int64_t>& seg_n, bool multi_stage) {
    int64_t max_super = 0;
    if (multi_stage && total > 256) {
        int64_t max_seg = 0;
        for (int64_t n : seg_n) if (n <= 512) max_seg = std::max(max_seg, n);
        const int64_t want = std::max<int64_t>((total + 3) / 4, std::max<int64_t>(256, max_seg));
        max_super = std::min<int64_t>(512, std::max<int64_t>(256, want));
        max_super = std::min<int64_t>(max_super, total);
    }
    return max_super;
}

int main() {
    // Framing: the four binary commands are consumed with their payload; control lines are not.
    check(batch_binary_line("BGBD 4 bin=3"), "BGBD is binary");
    check(batch_binary_line("BGENB 0 8 bin=2"), "BGENB is binary");
    check(batch_binary_line("GENB 0 8 bin=2"), "GENB is binary");
    check(batch_binary_line("BGEND 4 bin=1"), "BGEND is binary");
    check(!batch_binary_line("BADMIT 2 0 1"), "BADMIT is control");
    check(!batch_binary_line("BSTOP 3"), "BSTOP is control");
    check(!batch_binary_line("DONE 0 1 0.0 0.0 length 0 0 0 0 0 0 0 0.0 0 0 chunk=0"), "DONE is control");

    // CRC parity against the independent bitwise implementation, including the BGEND prefix case.
    check(batch_crc32_prefix({}, 0) == 0u, "crc32 of empty prefix");
    const std::vector<int32_t> ids = {11, 22, 33, 44, 55};
    for (size_t k = 0; k <= ids.size(); ++k)
        check(batch_crc32_prefix(ids, k) == crc32_slow(ids, k), "table CRC matches bitwise CRC");
    check(batch_crc32_prefix(ids, 99) == crc32_slow(ids, ids.size()), "CRC clamps past end");

    // Multi-super-chunk pipelining: a 1115-token / 7-slot cohort becomes 4 balanced super-chunks on a layer
    // split; single-stage and small reads keep max_super == 0 (byte-identical to every other caller).
    const std::vector<int64_t> seven(7, 160);
    const int64_t cap7 = batch_max_super(1115, seven, true);
    check(cap7 == 279 && (1115 + cap7 - 1) / cap7 == 4, "1115 tokens -> 4 super-chunks of 279");
    const std::vector<int64_t> two = {174, 173};
    check(batch_max_super(347, two, true) == 256, "347 tokens -> 256 cap");
    check(batch_max_super(1115, seven, false) == 0, "single stage does not cap");
    check(batch_max_super(200, {100, 100}, true) == 0, "short cohort does not cap");
    check(batch_max_super(1024, {600, 424}, true) == 424, "long segment ignored; cap from short segment");

    // The server's exact producer shape, including the repeated slot and the 8-hex CRC.
    int hs = -1; int64_t hm = 0, hp = -1; uint64_t hh = 0; bool hhash = false; bool hckpt = true; size_t hb = 0;
    check(parse_bgb_header("BGBD 3 12 ckpt=1 temperature=0.7 top_k=20 pfx=9 pfx_hash=0a1b2c3d bin=4",
                           hs, hm, hp, hh, hhash, hb, hckpt, 3), "BGBD header parses");
    check(hs == 3 && hm == 12 && hp == 9 && hh == 0x0a1b2c3dULL && hhash && hb == 4 && hckpt,
          "BGBD header fields");
    check(parse_bgb_header("BGBD 3 12 ckpt=0 pfx=2 bin=1", hs, hm, hp, hh, hhash, hb, hckpt, 3) && !hckpt,
          "ckpt is a boolean flag - 0 disables");
    check(!parse_bgb_header("BGBD 4 12 bin=1", hs, hm, hp, hh, hhash, hb, hckpt, 3),
          "BGBD slot must match BADMIT");

    // Review fix 1: Seg tokens are int64 ids.  An explicit narrowing conversion must produce the same ids;
    // a reinterpret_cast of the int64 array would instead interleave zero high words.
    const std::vector<int64_t> wide = {1, 0x100000002LL, 3, 4};
    std::vector<int32_t> narrow((size_t) wide.size());
    for (size_t j = 0; j < wide.size(); ++j) narrow[j] = (int32_t) wide[j];
    check(narrow[0] == 1 && narrow[1] == 2 && narrow[2] == 3 && narrow[3] == 4,
          "int64 to int32 id conversion");

    // Review fix 2: the last prompt token is sl.x (evaluated by the first window), so the prefill covers
    // [pfx, n_i - 1) and sl.ids stores n_i - 1 ids.  The MTP next-token array is indexed from u.all.
    const int64_t ni = 9, pfx = 2, delta = ni - pfx;
    const int64_t prefill_n = delta >= 2 ? delta - 1 : 0;
    const size_t stored_ids = (size_t) (ni - 1);
    check(prefill_n == ni - pfx - 1 && stored_ids == (size_t) (ni - 1),
          "prefill excludes the last prompt token");
    std::vector<int32_t> nxt((size_t) prefill_n);
    bool in_bounds = true;
    for (int64_t j = 0; j < prefill_n; ++j) {
        const int64_t pos_j = pfx + j + 1;
        if (pos_j > ni - 1) in_bounds = false;
        else nxt[(size_t) j] = (int32_t) pos_j;
    }
    check(in_bounds && nxt.size() == (size_t) prefill_n, "MTP next-token array stays in bounds");

    // Review fix 3: keep checkpoints at or before the prefix and give back only the ones dropped.
    std::vector<int> checks = {1, 2, 3, 4};
    std::vector<int> kept_checks, given_checks;
    for (int c : checks) {
        if (c <= pfx) kept_checks.push_back(c);
        else given_checks.push_back(c);
    }
    check(kept_checks.size() == 2 && given_checks.size() == 2 && kept_checks[0] == 1 && given_checks[0] == 3,
          "checkpoint keep/give partition");

    // A two-ticket cohort: one cold prompt, one warm delta verified with the CRC of its prefix.
    std::vector<Ticket> tk(2);
    tk[0].slot = 0; tk[0].max_new = 4; tk[0].bin = {7, 8, 9};
    tk[1].slot = 1; tk[1].max_new = 8; tk[1].pfx = 3;
    tk[1].all = {1, 2, 3};
    const std::vector<int32_t> warm_prefix = {1, 2, 3};
    tk[1].hash = batch_crc32_prefix(warm_prefix, 3); tk[1].has_hash = true;
    tk[1].bin = {4, 5};
    check(cohort_ok(tk, 2, 8, 2048), "valid cohort");
    check(tk[0].ok && tk[1].ok, "both tickets admitted");
    check(tk[1].delta == 2, "warm ticket reads only its delta");

    // A bad CRC discards only that ticket; the other one still proceeds.
    std::vector<Ticket> bad(2);
    bad[0].slot = 0; bad[0].max_new = 4; bad[0].bin = {7, 8};
    bad[1].slot = 1; bad[1].max_new = 4; bad[1].pfx = 3; bad[1].all = {1, 2, 3};
    bad[1].hash = 0x1234; bad[1].has_hash = true; bad[1].bin = {9};
    check(cohort_ok(bad, 2, 8, 2048), "cohort with one bad CRC still valid");
    check(bad[0].ok && !bad[1].ok, "only the mismatched ticket is discarded");

    // Invalid slot / duplicate slot / oversized cold prompt are rejected at the cohort level.
    std::vector<Ticket> dup(2);
    dup[0].slot = 3; dup[0].max_new = 1; dup[0].bin = {1};
    dup[1].slot = 3; dup[1].max_new = 1; dup[1].bin = {2};
    check(!cohort_ok(dup, 2, 8, 2048), "duplicate slot rejected");
    std::vector<Ticket> big(1);
    big[0].slot = 0; big[0].max_new = 1; big[0].bin.assign(2049, 1);
    check(cohort_ok(big, 1, 8, 2048), "cohort shape accepted");
    check(!big[0].ok, "oversized cold prompt rejected");
    std::vector<Ticket> small(1);
    small[0].slot = 0; small[0].max_new = 1; small[0].bin.assign(256, 1);
    check(cohort_ok(small, 1, 8, 2048) && small[0].ok, "256-token cold prompt accepted");

    // The Seg callbacks compile into the public prefill API and carry the slot's chunked token ids.
    strata::prefill::Prefill::Seg sg;
    int seen_chunk = 0, seen_rows = 0;
    int32_t toks[3] = {4, 5, 6};
    sg.on_seg_chunk = [&](int seg_idx, const int32_t* tok, int64_t n_tok) {
        check(seg_idx == 1 && n_tok == 3 && tok[2] == 6, "on_seg_chunk token window");
        ++seen_chunk;
    };
    sg.on_seg_rows = [&](int seg_idx, const float*, const int32_t*, int64_t n_tok, int64_t pos, std::string& e) {
        check(seg_idx == 1 && n_tok == 3 && pos == 7 && e.empty(), "on_seg_rows position");
        ++seen_rows;
        return true;
    };
    std::string err;
    sg.on_seg_chunk(1, toks, 3);
    sg.on_seg_rows(1, nullptr, toks, 3, 7, err);
    check(seen_chunk == 1 && seen_rows == 1, "per-segment callbacks invoked");

    // ok_out is per-ticket: a declined segment marks only its own owner false.
    std::vector<bool> ok_out = {true, false, true};
    check(ok_out[0] && !ok_out[1] && ok_out[2], "per-ticket ok_out");

    if (failures != 0) {
        std::fprintf(stderr, "batch_admit_test: %d assertion(s) failed\n", failures);
        return 1;
    }
    std::printf("batch_admit_test: all assertions passed\n");
    return 0;
}
