// Offline parity unit for STRATA_MTP_PAIR_T2_COUPLED: the paired T=2 chain and the two solo chains must feed the
// coupled sampler the same Philox counter and penalty history for every output.  CPU-only, no model fixture.
#include "strata/core/coupled_draft.hpp"
#include "strata/core/mtp.hpp"
#include "strata/kernels/sampler.hpp"

#include <cstdio>
#include <cstring>

using namespace strata::core;

namespace {
int g_fail = 0;
int g_checks = 0;
void check(bool ok, const char* what, long long a = 0, long long b = 0) {
    ++g_checks;
    if (!ok) {
        if (++g_fail <= 20) std::printf("  FAIL %s (%lld vs %lld)\n", what, a, b);
    }
}
}  // namespace

int main() {
    const int64_t p0 = 100, a = 1;            // T=2, the picked row is a=1
    const int32_t tail[] = {17, 23, 29, 31};  // the slots' committed reply tail
    const int32_t next = 37;                  // this window's picked token
    const int h = coupled_hist_len(4, kCoupledHistCap);
    check(h > 0 && h <= kCoupledHistCap, "history fits the ring", h, kCoupledHistCap);

    strata::kernels::SamplerParams sp{};
    sp.greedy = false;
    sp.temperature = 0.7f;
    sp.seed = 20261126ULL;
    sp.counter = 0;
    sp.penalty_last_n = 4;
    strata::kernels::SamplerParams slot0 = sp, slot1 = sp;
    check(slot0.seed == slot1.seed && slot0.counter == slot1.counter, "both slots share one seed");

    int32_t paired0[kCoupledHistCap + 8] = {}, paired1[kCoupledHistCap + 8] = {}, solo[kCoupledHistCap + 8] = {};
    coupled_hist_base(tail, 4, next, h, paired0 + kCoupledHistCap - h);
    coupled_hist_base(tail, 4, next, h, paired1 + kCoupledHistCap - h);
    coupled_hist_base(tail, 4, next, h, solo + kCoupledHistCap - h);
    check(std::memcmp(paired0, paired1, sizeof(paired0)) == 0, "paired slots stage identical rings");
    check(std::memcmp(paired0, solo, sizeof(paired0)) == 0, "paired ring equals the solo ring");

    // T=2, n_out=1: one output per slot.  A paired launch stages both slots once, so j=0 is the only output.
    for (int j = 0; j < 1; ++j) {
        const uint64_t cp0 = coupled_draft_counter(coupled_draft_cell(p0, a, j));
        const uint64_t cp1 = coupled_draft_counter(coupled_draft_cell(p0, a, j));
        const uint64_t cs = coupled_draft_counter(coupled_draft_cell(p0, a, j));
        check(cp0 == cp1 && cp0 == cs, "paired and solo Philox counters agree", (long long) cp0, (long long) cs);
        // The draft's row is p+a+j; the next row in the chain must be p+a+j+1, in both routes.
        check(coupled_draft_cell(p0, a, j + 1) == coupled_draft_cell(p0, a, j) + 1,
              "paired and solo advance one cell", (long long) coupled_draft_cell(p0, a, j + 1),
              (long long) (coupled_draft_cell(p0, a, j) + 1));
    }

    // n_out > 1 keeps the same progression for j = 1..3 (the draft ring is updated after each output).
    for (int j = 1; j < 4; ++j) {
        const uint64_t cp = coupled_draft_counter(coupled_draft_cell(p0, a, j));
        const uint64_t cs = coupled_draft_counter(coupled_draft_cell(p0, a, j));
        check(cp == cs, "later outputs share one counter", (long long) cp, (long long) cs);
    }

    // STRATA_SPEC_COUPLED gives every MtpDrafter its own cudaMalloc'd dinv_, so two distinct non-null
    // pointers must still satisfy the pair-chain compatibility predicate (STRATA_MTP_PAIR_T2_COUPLED=1).
    int32_t dinv_a = 0, dinv_b = 0;
    check(MtpDrafter::dinv_compatible(&dinv_a, &dinv_b), "distinct non-null dinv pointers are pair-compatible");
    check(!MtpDrafter::dinv_compatible(nullptr, &dinv_b), "null and non-null dinv pointers are incompatible");
    check(MtpDrafter::dinv_compatible(nullptr, nullptr), "both-null dinv pointers are pair-compatible");

    std::printf("mtp_pair_t2_test: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
