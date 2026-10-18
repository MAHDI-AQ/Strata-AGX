// src/kernels/cuda/verify_kernels.cu - see include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic of every kernel here is transcribed from its single-token original (fused_gdn.cu,
// elementwise.cu) with the same operation order, so a verify window reproduces plain decode bit for bit.
#include <unistd.h>
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/dp4a.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); _exit(1); }
}

__global__ void __launch_bounds__(S) gdn_conv_l2_multi_kernel(const float* __restrict__ hist,
                                                              const float* __restrict__ qkv,
                                                              const float* __restrict__ w, float* __restrict__ h,
                                                              int C, int qk_heads, float eps, int t_begin) {
    __shared__ float part[S / 32];
    const int t = t_begin + blockIdx.y;
    const int c = blockIdx.x * S + threadIdx.x;
    // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
    float win[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = t + j;          // index into [hist(3) | x...]
        win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
    float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];
    float y = sum / (1.0f + __expf(-sum));
    if ((int) blockIdx.x < qk_heads) {
        float sq = y * y;
        for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
        if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
        __syncthreads();
        const float ss = part[0] + part[1] + part[2] + part[3];
        y *= rsqrtf(ss + eps);
    }
    h[(size_t) t * C + c] = y;
}

__global__ void gdn_conv_commit_kernel(float* __restrict__ hist, const float* __restrict__ qkv, int C,
                                       const int32_t* __restrict__ n_keep) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int n = *n_keep;
    if (n <= 0) return;
    float seq[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int src = n + j;          // the last three of [hist(3) | x_0..x_{n-1}]
        seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
    }
    hist[c * 3] = seq[0];
    hist[c * 3 + 1] = seq[1];
    hist[c * 3 + 2] = seq[2];
}

__global__ void __launch_bounds__(256) gdn_ab_multi_kernel(const float* __restrict__ x, const uint16_t* __restrict__ wa,
                                                           const uint16_t* __restrict__ wb,
                                                           const float* __restrict__ dt,
                                                           const float* __restrict__ ssm_a, float* __restrict__ gate,
                                                           float* __restrict__ beta, int n, int h_v, int T) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= 2 * h_v) return;
    const bool is_beta = row >= h_v;
    const int r = is_beta ? row - h_v : row;
    const uint4* w4 = reinterpret_cast<const uint4*>((is_beta ? wb : wa) + (size_t) r * n);
    float acc[kVerifyMaxT];
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) acc[t] = 0.0f;
    for (int j = lane; j < n / 8; j += 32) {
        const uint4 wv = __ldg(w4 + j);
#pragma unroll
        for (int t = 0; t < kVerifyMaxT; ++t) {
            if (t >= T) break;
            const float* xt = x + (size_t) t * n;
            const float4 xa = *reinterpret_cast<const float4*>(xt + j * 8);
            const float4 xb = *reinterpret_cast<const float4*>(xt + j * 8 + 4);
            float a = acc[t];
            a = fmaf(__uint_as_float(wv.x << 16), xa.x, a); a = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, a);
            a = fmaf(__uint_as_float(wv.y << 16), xa.z, a); a = fmaf(__uint_as_float(wv.y & 0xffff0000u), xa.w, a);
            a = fmaf(__uint_as_float(wv.z << 16), xb.x, a); a = fmaf(__uint_as_float(wv.z & 0xffff0000u), xb.y, a);
            a = fmaf(__uint_as_float(wv.w << 16), xb.z, a); a = fmaf(__uint_as_float(wv.w & 0xffff0000u), xb.w, a);
            acc[t] = a;
        }
    }
#pragma unroll
    for (int t = 0; t < kVerifyMaxT; ++t) {
        if (t >= T) break;
        float a = acc[t];
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane != 0) continue;
        if (is_beta) {
            beta[(size_t) t * h_v + r] = 1.0f / (1.0f + __expf(-a));
        } else {
            const float v = a + dt[r];
            const float sp = v > 20.0f ? v : log1pf(__expf(v));
            gate[(size_t) t * h_v + r] = sp * ssm_a[r];
        }
    }
}

__global__ void __launch_bounds__(S * RG) gdn_step_norm_multi_kernel(float* __restrict__ state,
                                                                     const float* __restrict__ hbuf, int C,
                                                                     const float* __restrict__ gate,
                                                                     const float* __restrict__ beta,
                                                                     const float* __restrict__ z,
                                                                     const float* __restrict__ gamma, float eps,
                                                                     float* __restrict__ y, int h_k, int h_v, int T,
                                                                     const int32_t* __restrict__ n_keep, int t_out_begin) {
    __shared__ float sk[S], sq[S];
    __shared__ float red[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const int rg = threadIdx.y;
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int qk = S * h_k;             // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
    const int value_dim = S * h_v;
    const int n = n_keep ? *n_keep : T;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < n; ++t) {
        const float* ht = hbuf + (size_t) t * C;
        __syncthreads();                // the previous token is done with sk/sq/red/wsum
        if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red[rg][col] = kv;
        __syncthreads();
        const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];
        const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        __syncthreads();
        red[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * value_dim + head * S + col];
            y[(size_t) t * value_dim + head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));
        }
    }
    if (n_keep != nullptr && n > 0) {
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
    }
}

__global__ void embedding_gather_dev_kernel(const uint8_t* __restrict__ codes, const float* __restrict__ scales,
                                            const float* __restrict__ offsets, const int32_t* __restrict__ tokens,
                                            int64_t n, int code_bits, int code_bias, int group_elems,
                                            unsigned long long row_codes, unsigned long long row_groups,
                                            float* __restrict__ out) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long token = (unsigned long long) tokens[t];
    const uint8_t* c = codes + token * row_codes;
    const float* sc = scales + token * row_groups;
    const float* of = offsets ? offsets + token * row_groups : nullptr;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = __fmul_rn((float) (code + code_bias), sc[group]);
    out[(size_t) t * n + i] = __fadd_rn(product, of ? of[group] : 0.0f);
}

__global__ void broadcast_streams_kernel(const float* __restrict__ x, float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = x[(size_t) t * n + i % n];
}

__global__ void copy_indexed_kernel(float* __restrict__ dst, const float* __restrict__ src, int64_t stride,
                                    const int32_t* __restrict__ index, int64_t n) {
    const int idx = *index;
    if (idx < 0) return;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        dst[i] = src[(size_t) idx * stride + i];
}

__global__ void fetch_blobs_kernel(const unsigned long long* __restrict__ src, const int32_t* __restrict__ n,
                                   uint4* __restrict__ dst, long long per) {
    const long long total = (long long) *n * per;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long) gridDim.x * blockDim.x) {
        const long long k = i / per, off = i - k * per;
        dst[i] = ((const uint4*) src[k])[off];
    }
}

__global__ void rebase_ptrs_kernel(unsigned long long* ptr, const int32_t* n, unsigned long long base, long long bytes) {
    const int k = threadIdx.x;
    if (k < *n) ptr[k] = base + (unsigned long long) k * (unsigned long long) bytes;
}

__global__ void add_streams_broadcast_kernel(const float* __restrict__ h, const float* __restrict__ e,
                                             float* __restrict__ R, int64_t n, int hc) {
    const int t = blockIdx.y;
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * hc) return;
    R[(size_t) t * n * hc + i] = h[(size_t) t * n * hc + i] + e[(size_t) t * n + i % n];
}

__global__ void ident_hits_kernel(const int32_t* __restrict__ ids, int n, int32_t* __restrict__ slot,
                                  int32_t* __restrict__ dst, int32_t* __restrict__ count) {
    const int i = threadIdx.x;
    if (i < n) { slot[i] = ids[i]; dst[i] = i; }
    if (i == 0) *count = n;
}

// E = the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
template<typename E>
__global__ void gather_rows_kernel(const E* __restrict__ src, long long row_e, const int32_t* __restrict__ ids,
                                   long long n, E* __restrict__ dst) {
    const long long total = n * row_e;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (long long) gridDim.x * blockDim.x) {
        const long long r = i / row_e, o = i - r * row_e;
        dst[i] = src[(long long) ids[r] * row_e + o];
    }
}

__global__ void map_ids_kernel(int32_t* ids, const int32_t* __restrict__ table, int n) {
    const int i = threadIdx.x;
    if (i < n) ids[i] = table[ids[i]];
}

__global__ void row_top_prob_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                                    float* __restrict__ probs, float inv_temp = 1.0f) {
    __shared__ float part[32];
    const int t = blockIdx.x;
    const float* l = logits + (size_t) t * n_vocab;
    const float m = l[ids[t]];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_vocab; i += blockDim.x) {
        const float diff = (l[i] - m) * inv_temp;
        s += (diff > -80.0f) ? __expf(diff) : 0.0f;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.0f;
        for (int w = 0; w < (int) (blockDim.x >> 5); ++w) tot += part[w];
        probs[t] = tot > 0.0f ? 1.0f / tot : 0.0f;
    }
}

__global__ void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride, const int32_t* __restrict__ ids,
                                  const int32_t* __restrict__ row_dev, float* __restrict__ R_dst,
                                  int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p) {
    const int row = *row_dev;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += (int64_t) gridDim.x * blockDim.x)
        R_dst[i] = R_src[(size_t) row * stride + i];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
    }
}

__global__ void dense_steps_kernel(const int32_t* __restrict__ cells, int n, int32_t* __restrict__ steps) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int c = cells[i];
    steps[i * 4 + 0] = c;
    steps[i * 4 + 1] = c + 1;
    steps[i * 4 + 2] = (c + 1) / 4;
    steps[i * 4 + 3] = c + 1;
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); _exit(1); }
    fetch_blobs_kernel<<<48 * 8, 256, 0, (cudaStream_t) stream>>>(src, n, (uint4*) dst, (long long) (blob_bytes / 16));
    check("fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    rebase_ptrs_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(ptr, n, (unsigned long long) base, (long long) blob_bytes);
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    add_streams_broadcast_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                                   (cudaStream_t) stream>>>(h, e, R, n_embd, hc);
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); _exit(1); }
    ident_hits_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(ids, n, slot, dst, count);
    check("ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    mtp_select_kernel<<<16, 256, 0, (cudaStream_t) stream>>>(R_src, R_stride, ids, row_dev, R_dst, tok_dst, out, j,
                                                             probs, out_p);
    check("mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    if (row_bytes % 16 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint4*) src, row_bytes / 16, ids, n, (uint4*) dst);
    else if (row_bytes % 4 == 0)
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>((const uint32_t*) src, row_bytes / 4, ids, n, (uint32_t*) dst);
    else
        gather_rows_kernel<<<48 * 8, 256, 0, s>>>(src, row_bytes, ids, n, dst);
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    map_ids_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(ids, table, n);
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream, float inv_temp) {
    row_top_prob_kernel<<<n_rows, 1024, 0, (cudaStream_t) stream>>>(logits, n_vocab, ids, probs, inv_temp);
    check("row_top_prob");
}

#if !defined(__HIPCC__)   // warp-32 kernels: an AMD card keeps the one-block row_top_prob and sampler_greedy_kernel
namespace {

constexpr int kArgMaxBlocks = 128, kArgThreads = 256, kTopBlocks = 8;

// the larger value, on equality the lower index: sampler_greedy_kernel's order
__device__ __forceinline__ void arg_take(float ov, int oi, float& bv, int& bi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// a block's (value, index) pick, in thread 0; `none` is the index of no candidate
__device__ __forceinline__ void arg_block(float& bv, int& bi, int none) {
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        arg_take(ov, oi, bv, bi);
    }
    const int warp = (int) (threadIdx.x >> 5), lane = (int) (threadIdx.x & 31);
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (warp == 0) {
        const int nw = (int) (blockDim.x >> 5);
        bv = lane < nw ? sv[lane] : __int_as_float(0xff800000);
        bi = lane < nw ? si[lane] : none;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oi = __shfl_down_sync(0xffffffffu, bi, off);
            arg_take(ov, oi, bv, bi);
        }
    }
}

// A row's scratch, whatever the launch's row count (one scratch serves windows of every size): its counter, then
// the blocks' values and indices.
struct ArgRow {
    unsigned counter;
    unsigned pad[3];
    float v[kArgMaxBlocks];
    int i[kArgMaxBlocks];
};

// grid (blocks, rows): block b scans [b * per_block, +per_block) of its row
__global__ void argmax_rows_kernel(const float* __restrict__ logits, int n, int per_block, ArgRow* __restrict__ rows,
                                   int32_t* out) {
    const int row = (int) blockIdx.y, b = (int) blockIdx.x, nb = (int) gridDim.x;
    unsigned* counter = &rows[row].counter;
    float* pv = rows[row].v;
    int* pi = rows[row].i;
    const float* l = logits + (size_t) row * n;
    const int lo = b * per_block, hi = min(n, lo + per_block);
    float bv = __int_as_float(0xff800000);   // -inf
    int bi = n;
    for (int v = lo + (int) threadIdx.x; v < hi; v += (int) blockDim.x) {
        const float s = l[v];
        if (s > bv) { bv = s; bi = v; }
    }
    arg_block(bv, bi, n);
    __shared__ bool last;
    if (threadIdx.x == 0) {
        pv[b] = bv;
        pi[b] = bi;
        __threadfence();
        last = atomicAdd(counter, 1u) == (unsigned) (nb - 1);
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    bv = __int_as_float(0xff800000);
    bi = n;
    if ((int) threadIdx.x < nb) {
        bv = ((volatile const float*) pv)[threadIdx.x];
        bi = ((volatile const int*) pi)[threadIdx.x];
    }
    arg_block(bv, bi, n);
    if (threadIdx.x == 0) {
        out[row] = bi < n ? bi : 0;   // no value above -inf: 0, as the one-block kernel
        *counter = 0;
    }
}

// a row's top-probability scratch: its counter, then its 32 warps' sums
struct TopRow {
    unsigned counter;
    unsigned pad[3];
    float parts[32];
};

// grid (kTopBlocks, rows), 1024 / kTopBlocks threads: thread j of block b is row_top_prob's thread b * blockDim + j
__global__ void row_top_prob_split_kernel(const float* __restrict__ logits, int n_vocab, const int32_t* __restrict__ ids,
                                          float* __restrict__ probs, TopRow* __restrict__ rows) {
    const int row = (int) blockIdx.y;
    float* parts = rows[row].parts;
    const float* l = logits + (size_t) row * n_vocab;
    const float m = l[ids[row]];
    const int vt = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    float s = 0.0f;
    for (int i = vt; i < n_vocab; i += 1024) {
        const float diff = l[i] - m;
        s += (diff > -80.0f) ? __expf(diff) : 0.0f;
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) parts[vt >> 5] = s;
    __threadfence();
    __syncthreads();
    __shared__ bool last;
    if (threadIdx.x == 0) last = atomicAdd(&rows[row].counter, 1u) == (unsigned) (gridDim.x - 1);
    __syncthreads();
    if (!last || threadIdx.x != 0) return;
    __threadfence();
    float tot = 0.0f;
    for (int w = 0; w < 32; ++w) tot += ((volatile const float*) parts)[w];
    probs[row] = tot > 0.0f ? 1.0f / tot : 0.0f;
    rows[row].counter = 0;
}
}  // namespace

uint64_t argmax_rows_scratch_bytes(int n_rows) { return (uint64_t) n_rows * sizeof(ArgRow); }

void argmax_rows(const float* logits, int n_rows, int n, void* scratch, int32_t* out, void* stream) {
    if (n_rows <= 0) return;
    const int nb = std::min(kArgMaxBlocks, std::max(1, (n + 4095) / 4096));
    const int per_block = (n + nb - 1) / nb;
    argmax_rows_kernel<<<dim3((unsigned) nb, (unsigned) n_rows), kArgThreads, 0, (cudaStream_t) stream>>>(
        logits, n, per_block, (ArgRow*) scratch, out);
    check("argmax_rows");
}

uint64_t row_top_prob_scratch_bytes(int n_rows) { return (uint64_t) n_rows * sizeof(TopRow); }

void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* scratch,
                        void* stream) {
    if (n_rows <= 0) return;
    row_top_prob_split_kernel<<<dim3(kTopBlocks, (unsigned) n_rows), 1024 / kTopBlocks, 0, (cudaStream_t) stream>>>(
        logits, n_vocab, ids, probs, (TopRow*) scratch);
    check("row_top_prob_split");
}
#else
uint64_t argmax_rows_scratch_bytes(int) { return 0; }
void argmax_rows(const float*, int, int, void*, int32_t*, void*) {
    std::fprintf(stderr, "argmax_rows: not built for this backend\n");
    _exit(1);
}
uint64_t row_top_prob_scratch_bytes(int) { return 0; }
void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void*,
                        void* stream) {
    row_top_prob(logits, n_rows, n_vocab, ids, probs, stream);
}
#endif

bool multi_block_head_ops() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MULTI_BLOCK_ARGMAX");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return on;
#endif
}

bool argmax_rows_wanted() {
#if defined(__HIPCC__)
    return false;
#else
    // sm_80 to sm_89: sample_tokens' greedy pick there is the one-block kernel (sm_90+ has the cluster kernel)
    static const bool want = [] {
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
            return false;
        return major == 8;
    }();
    return want && multi_block_head_ops();
#endif
}

namespace {
__global__ void window_ids_kernel(int32_t* steps, int window, int32_t* ids, long long stride) {
    const int q = blockIdx.y;
    int32_t* st = steps + q * 4;
    const int n_kv = st[1];
    const int start = n_kv > window ? n_kv - window : 0;
    const int width = n_kv - start;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < width; j += gridDim.x * blockDim.x)
        ids[q * stride + j] = start + j;
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) st[3] = width;
}
}  // namespace

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    window_ids_kernel<<<dim3(8, (unsigned) n), 256, 0, (cudaStream_t) stream>>>(steps, window, ids, (long long) ids_stride);
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    dense_steps_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(cells, n, steps);
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        _exit(1);
    }
    gdn_conv_l2_multi_kernel<<<dim3((unsigned) (channels / S), (unsigned) n_tok), S, 0, (cudaStream_t) stream>>>(
        history, qkv, conv_w, h, channels, qk_heads, eps, t_begin);
    check("gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    gdn_conv_commit_kernel<<<(unsigned) ((channels + 255) / 256), 256, 0, (cudaStream_t) stream>>>(history, qkv,
                                                                                                 channels, n_keep);
    check("gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        _exit(1);
    }
    gdn_ab_multi_kernel<<<(unsigned) ((2 * h_v + 7) / 8), 256, 0, (cudaStream_t) stream>>>(
        x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
    check("gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !h || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        _exit(1);
    }
    gdn_step_norm_multi_kernel<<<(unsigned) h_v, dim3(S, RG), 0, (cudaStream_t) stream>>>(
        state, h, conv_channels, gate, beta, z, gamma, eps, y, h_k, h_v, n_tok, n_keep, t_out_begin);
    check("gdn_step_norm_multi");
}

namespace {
__global__ void wait_flag_ge_kernel(const volatile uint32_t* flag, uint32_t value) {
    // HW-01: Adaptive backoff during graph doorbell polling to avoid saturating PCIe bus
    int spin = 0;
    while (*flag < value) {
        if (++spin < 64) {
            strata_spin_pause();
        } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(1000);
#else
            strata_spin_pause();
#endif
        }
    }
    __threadfence_system();
}
}  // namespace

namespace {
__global__ void resident_plan_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                     int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                     long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                     uint32_t ring) {
    // one thread: at most kVerifyMaxT * 10 entries, the host's exact loop
    for (int i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || e >= n_expert || res[e] < 0) { *skip = 0; return; }
    }
    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;
    int groups = 0, entries = 0;
    for (int i0 = 0; i0 < n; ++i0) {
        bool first = true;
        for (int j = 0; j < i0; ++j) if (ids[j] == ids[i0]) { first = false; break; }
        if (!first) continue;
        const int32_t slot = res[ids[i0]];
        ptr[groups] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
        start[groups] = entries;
        for (int i = i0; i < n; ++i)
            if (ids[i] == ids[i0]) {
                // an entry belongs to i0's group when its first occurrence is i0: the same expert id
                dst[entries] = i;
                tok[entries] = i / k;
                ++entries;
            }
        ++groups;
    }
    start[groups] = entries;
    start2[0] = entries;
    counts[0] = groups;
    counts[1] = entries;
    counts[2] = 0;
    __threadfence();
    *skip = ring;
}
__global__ void wait_flag_ge_or_kernel(const volatile uint32_t* flag, uint32_t value, const volatile uint32_t* skip) {
    if (*skip == value) return;
    // HW-01: Adaptive backoff during graph doorbell polling to avoid saturating PCIe bus
    int spin = 0;
    while (*flag < value) {
        if (++spin < 64) {
            strata_spin_pause();
        } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            __nanosleep(1000);
#else
            strata_spin_pause();
#endif
        }
    }
    __threadfence_system();
}
__global__ void copy_i32_unless_kernel(int32_t* __restrict__ dst, const volatile int32_t* src, int n,
                                       const uint32_t* skip, uint32_t value) {
    if (*skip == value) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
__global__ void copy_or_zero_kernel(float4* __restrict__ dst, const volatile float4* src, long long n4,
                                    const uint32_t* skip, uint32_t value) {
    const bool zero = *skip == value;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long) gridDim.x * blockDim.x)
        dst[i] = zero ? make_float4(0.f, 0.f, 0.f, 0.f) : const_cast<const float4*>(src)[i];
}
__global__ void resident_plan_warp_kernel(const int32_t* __restrict__ ids, int n, int k, const int32_t* __restrict__ res,
                                          int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off,
                                          long long blob, int32_t* __restrict__ pl, long long capx, uint32_t* skip,
                                          uint32_t ring) {
    const int tid = threadIdx.x;
    __shared__ int32_t s_ids[512];

    int invalid = 0;
    for (int i = tid; i < n; i += 32) {
        const int32_t e = ids[i];
        if (i < 512) s_ids[i] = e;
        if (e < 0 || e >= n_expert || res[e] < 0) invalid = 1;
    }
    if (__any_sync(0xffffffff, invalid)) {
        if (tid == 0) *skip = 0;
        return;
    }
    __syncwarp();

    int32_t* counts = pl;
    int32_t* start = pl + 4;
    int32_t* dst = start + capx + 1;
    int32_t* tok = dst + capx;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    unsigned long long* ptr = (unsigned long long*) (pl + ptr_off);
    int32_t* start2 = pl + ptr_off + 4 * capx;

    int groups = 0, entries = 0;

    for (int i0 = 0; i0 < n; ++i0) {
        const int32_t target_e = (i0 < 512) ? s_ids[i0] : ids[i0];
        bool first = true;
        for (int j = 0; j < i0; ++j) {
            const int32_t prev_e = (j < 512) ? s_ids[j] : ids[j];
            if (prev_e == target_e) { first = false; break; }
        }
        if (!first) continue;

        if (tid == 0) {
            const int32_t slot = res[target_e];
            ptr[groups] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
            start[groups] = entries;
        }

        for (int base = i0; base < n; base += 32) {
            const int idx = base + tid;
            const bool match = (idx < n) && (((idx < 512) ? s_ids[idx] : ids[idx]) == target_e);
            const uint32_t mask = __ballot_sync(0xffffffff, match);
            if (match) {
                const int rank = __popc(mask & ((1u << tid) - 1));
                dst[entries + rank] = idx;
                tok[entries + rank] = idx / k;
            }
            entries += __popc(mask);
        }
        ++groups;
    }

    if (tid == 0) {
        start[groups] = entries;
        start2[0] = entries;
        counts[0] = groups;
        counts[1] = entries;
        counts[2] = 0;
        __threadfence();
        *skip = ring;
    }
}
}  // namespace

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream) {
    static const bool use_warp = [] {
        const char* e = std::getenv("STRATA_WARP_PLAN");
        return e == nullptr || std::atoi(e) != 0;
    }();
    if (use_warp) {
        resident_plan_warp_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(ids, n_entries, k, res_layer, n_expert, cache_base,
                                                                       slot_off, blob, plan, capx, skip, ring);
    } else {
        resident_plan_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ids, n_entries, k, res_layer, n_expert, cache_base, slot_off,
                                                                 blob, plan, capx, skip, ring);
    }
    check("resident_plan");
}
void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    wait_flag_ge_or_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value, skip);
    check("wait_flag_ge_or");
}
void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    if (n > 0x7fffffffLL) {   // D2 bounds: the kernel's n is int; fail closed, never truncate
        std::fprintf(stderr, "copy_i32_from_mapped_unless: n=%lld exceeds the kernel bound; refusing\n", n);
        _exit(1);
    }
    copy_i32_unless_kernel<<<1, 128, 0, (cudaStream_t) stream>>>(dst, (const volatile int32_t*) src, (int) n, skip, value);
    check("copy_i32_from_mapped_unless");
}
void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    copy_or_zero_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>((float4*) dst, (const volatile float4*) src, n4, skip,
                                                                    value);
    check("copy_or_zero_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    wait_flag_ge_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(flag, value);
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    embedding_gather_dev_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) n_tok), 256, 0,
                                  (cudaStream_t) stream>>>(codes, scales, offsets, tokens, n, code_bits, code_bias,
                                                           group_elems, row_codes, row_groups, out);
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    broadcast_streams_kernel<<<dim3((unsigned) ((n_embd * hc + 255) / 256), (unsigned) n_tok), 256, 0,
                               (cudaStream_t) stream>>>(x, R, n_embd, hc);
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    copy_indexed_kernel<<<blocks, 256, 0, (cudaStream_t) stream>>>(dst, src, stride, index, n);
    check("copy_indexed");
}

// a GPU timestamp (ns, %globaltimer) into buf[i] - the verify window's stage profiler
namespace { __global__ void gpu_stamp_kernel(unsigned long long* buf, int i) {
    unsigned long long t;
#if defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter, in ns
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    buf[i] = t;
} }
void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    gpu_stamp_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(buf, i);
}

// Task 7.2: Device-Side Speculative Tree Verification & Path Selection Kernel
// Cross-Engine Reference: Sequoia (SOSP 2024), EAGLE-2 (arXiv:2406.16858)
namespace {
// Frontier 4: Warp-level MTP Tree Branch Selection Kernel
__global__ void mtp_tree_branch_select_kernel(
    const float* __restrict__ logits,
    int vocab_size,
    const int32_t* __restrict__ anchor_token,
    int64_t base_pos,
    int32_t* __restrict__ out_tree_tokens,
    int32_t* __restrict__ out_parents,
    int32_t* __restrict__ out_depths,
    int64_t* __restrict__ out_positions,
    float* __restrict__ out_log_probs) {

    // Thread 0 initializes root node
    if (threadIdx.x == 0) {
        int32_t root_tok = anchor_token ? *anchor_token : 0;
        out_tree_tokens[0] = root_tok;
        out_parents[0] = -1;
        out_depths[0] = 0;
        out_positions[0] = base_pos;
        out_log_probs[0] = 0.0f;
    }

    // Warp-level Top-2 reduction over vocab logits
    float max1 = -1e30f;
    int32_t id1 = 0;
    float max2 = -1e30f;
    int32_t id2 = 0;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = logits[i];
        if (val > max1) {
            max2 = max1; id2 = id1;
            max1 = val; id1 = i;
        } else if (val > max2) {
            max2 = val; id2 = i;
        }
    }

    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        float other1 = __shfl_down_sync(0xffffffff, max1, offset);
        int32_t other_id1 = __shfl_down_sync(0xffffffff, id1, offset);
        float other2 = __shfl_down_sync(0xffffffff, max2, offset);
        int32_t other_id2 = __shfl_down_sync(0xffffffff, id2, offset);

        if (other1 > max1) {
            max2 = (max1 > other2) ? max1 : other2;
            id2 = (max1 > other2) ? id1 : other_id2;
            max1 = other1;
            id1 = other_id1;
        } else if (other1 > max2) {
            max2 = other1;
            id2 = other_id1;
        } else if (other2 > max2) {
            max2 = other2;
            id2 = other_id2;
        }
    }

    if (threadIdx.x == 0) {
        // Node 1: Top-1 candidate
        out_tree_tokens[1] = id1;
        out_parents[1] = 0;
        out_depths[1] = 1;
        out_positions[1] = base_pos + 1;
        out_log_probs[1] = max1;

        // Node 2: Top-2 candidate
        out_tree_tokens[2] = id2;
        out_parents[2] = 0;
        out_depths[2] = 1;
        out_positions[2] = base_pos + 1;
        out_log_probs[2] = max2;
    }
}

__global__ void spec_tree_verify_dp_kernel(
    const int32_t* __restrict__ tree_tokens,
    const int32_t* __restrict__ target_tokens,
    const int32_t* __restrict__ parents,
    const float* __restrict__ log_probs,
    int32_t* __restrict__ best_path_indices,
    int32_t* __restrict__ n_accepted,
    int T) {

    __shared__ int32_t s_valid[16];
    __shared__ int32_t s_len[16];
    __shared__ float s_score[16];
    __shared__ int32_t s_parent[16];

    const int tid = threadIdx.x;
    if (tid < 16) {
        s_valid[tid] = 0;
        s_len[tid] = 0;
        s_score[tid] = -1e9f;
        s_parent[tid] = (tid < T) ? parents[tid] : -1;
    }
    __syncthreads();

    if (tid == 0) {
        if (T <= 0) {
            *n_accepted = 0;
            return;
        }

        // Root (node 0) is unconditionally valid as the prefix/anchor
        s_valid[0] = 1;
        s_len[0] = 1;
        s_score[0] = (log_probs != nullptr) ? log_probs[0] : 0.0f;

        // DP pass over nodes in topological order (parent[i] < i)
        for (int i = 1; i < T; ++i) {
            const int p = s_parent[i];
            if (p >= 0 && p < i && s_valid[p]) {
                const int32_t expected = target_tokens[p];
                const int32_t actual = tree_tokens[i];
                if (actual == expected) {
                    s_valid[i] = 1;
                    s_len[i] = s_len[p] + 1;
                    const float score_inc = (log_probs != nullptr) ? log_probs[i] : 0.0f;
                    s_score[i] = s_score[p] + score_inc;
                }
            }
        }

        // Find leaf node with maximum length, breaking ties with cumulative score
        int best_node = 0;
        int max_len = 1;
        float max_score = s_score[0];

        for (int i = 1; i < T; ++i) {
            if (s_valid[i]) {
                if (s_len[i] > max_len || (s_len[i] == max_len && s_score[i] > max_score)) {
                    max_len = s_len[i];
                    max_score = s_score[i];
                    best_node = i;
                }
            }
        }

        // Backtrack path from best_node to root
        int32_t path_rev[16];
        int curr = best_node;
        int count = 0;
        while (curr >= 0 && count < 16) {
            path_rev[count++] = curr;
            curr = s_parent[curr];
        }

        // Emit best_path_indices in forward (root-to-leaf) order
        for (int k = 0; k < count; ++k) {
            best_path_indices[k] = path_rev[count - 1 - k];
        }
        *n_accepted = count;
    }
}
__global__ void gpu_timeline_signal_kernel(uint64_t* timeline, uint64_t val) {
    if (threadIdx.x == 0 && timeline != nullptr) {
        __threadfence_system();
        atomicMax((unsigned long long*) timeline, (unsigned long long) val);
        __threadfence_system();
    }
}

__global__ void gpu_timeline_wait_ge_kernel(const volatile uint64_t* timeline, uint64_t val) {
    if (threadIdx.x == 0 && timeline != nullptr) {
        while (*timeline < val) {
            __nanosleep(50);
        }
        __threadfence_system();
    }
}

__global__ void gpu_stamp_kernel(uint32_t* flag, uint32_t val) {
    if (threadIdx.x == 0 && flag != nullptr) {
        __threadfence_system();
        atomicExch((unsigned int*) flag, (unsigned int) val);
        __threadfence_system();
    }
}

}  // namespace

void mtp_tree_branch_select(const float* logits,
                            int vocab_size,
                            const int32_t* anchor_token,
                            int64_t base_pos,
                            int32_t* out_tree_tokens,
                            int32_t* out_parents,
                            int32_t* out_depths,
                            int64_t* out_positions,
                            float* out_log_probs,
                            void* stream) {
    if (!logits || vocab_size <= 0) return;
    mtp_tree_branch_select_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(
        logits, vocab_size, anchor_token, base_pos,
        out_tree_tokens, out_parents, out_depths, out_positions, out_log_probs);
    check("mtp_tree_branch_select");
}

void spec_tree_verify_dp(const int32_t* tree_tokens,
                         const int32_t* target_tokens,
                         const int32_t* parents,
                         const float* log_probs,
                         int32_t* best_path_indices,
                         int32_t* n_accepted,
                         int T,
                         void* stream) {
    if (T <= 0) return;
    const int actual_t = (T > 16) ? 16 : T;
    spec_tree_verify_dp_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(
        tree_tokens, target_tokens, parents, log_probs, best_path_indices, n_accepted, actual_t);
    check("spec_tree_verify_dp");
}

void gpu_timeline_signal(uint64_t* timeline, uint64_t val, void* stream) {
    if (!timeline) return;
    gpu_timeline_signal_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(timeline, val);
    check("gpu_timeline_signal");
}

void gpu_timeline_wait_ge(const uint64_t* timeline, uint64_t val, void* stream) {
    if (!timeline) return;
    gpu_timeline_wait_ge_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(timeline, val);
    check("gpu_timeline_wait_ge");
}

void gpu_stamp(uint32_t* flag, uint32_t val, void* stream) {
    if (!flag) return;
    gpu_stamp_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(flag, val);
    check("gpu_stamp");
}

}  // namespace strata::kernels
