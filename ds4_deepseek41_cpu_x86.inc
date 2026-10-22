/* V4.1 CPU projections: SIMD lanes are output rows, never reduction terms.
 * This preserves the reference's rounding after every multiply and addition. */
#ifndef DS4_DEEPSEEK41_CPU_X86_INC
#define DS4_DEEPSEEK41_CPU_X86_INC

#if defined(__clang__)
#pragma float_control(precise, on, push)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>

/* Match the software conversion even for signaling NaNs and their payloads.
 * Mask special values before VCVTPH2PS so it does not quiet them prematurely. */
static inline __m512 ds41_cpu_half16(__m512i bits) {
    const __mmask16 special = _mm512_cmpeq_epi32_mask(
        _mm512_and_si512(bits, _mm512_set1_epi32(0x7c00)),
        _mm512_set1_epi32(0x7c00));
    const __m512 converted = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(
        _mm512_maskz_mov_epi32((__mmask16)~special, bits)));
    const __m512i exceptional = _mm512_or_si512(_mm512_set1_epi32(0x7f800000),
        _mm512_or_si512(
            _mm512_slli_epi32(_mm512_and_si512(bits, _mm512_set1_epi32(0x8000)), 16),
            _mm512_slli_epi32(_mm512_and_si512(bits, _mm512_set1_epi32(0x03ff)), 13)));
    return _mm512_mask_mov_ps(converted, special, _mm512_castsi512_ps(exceptional));
}

static inline __m512i ds41_cpu_row_offsets(uint64_t row_bytes) {
    return _mm512_mullo_epi32(_mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        _mm512_set1_epi32((int)row_bytes));
}

/* Each 32-bit gather reads four bytes within a complete stored quant block.
 * Q8 blocks are 34 bytes and may be unaligned. The last activation block may
 * be partial, but its stored quant block still has all 32 quant bytes. */
static __m512 ds41_cpu_q8_rows16(const uint8_t *rows, const float *x,
                                uint64_t in_dim, uint64_t row_bytes) {
    const __m512i offsets = ds41_cpu_row_offsets(row_bytes);
    __m512 sum = _mm512_setzero_ps();
    for (uint64_t i0 = 0, b = 0; i0 < in_dim; i0 += 32, b++) {
        const uint8_t *block = rows + b * 34;
        const __m512 d = ds41_cpu_half16(_mm512_i32gather_epi32(offsets, block, 1));
        const unsigned n = (unsigned)(in_dim - i0 < 32 ? in_dim - i0 : 32);
        for (unsigned i = 0; i < n; i += 4) {
            const __m512i packed = _mm512_i32gather_epi32(offsets, block + 2 + i, 1);
#define DS41_Q8_TERM(J) do { \
    const __m512 q = _mm512_cvtepi32_ps(_mm512_srai_epi32( \
        _mm512_slli_epi32(packed, 24 - 8 * (J)), 24)); \
    sum = _mm512_add_ps(sum, _mm512_mul_ps(_mm512_mul_ps(d, q), \
        _mm512_set1_ps(x[i0 + i + (J)]))); \
} while (0)
            DS41_Q8_TERM(0);
            if (i + 1 < n) DS41_Q8_TERM(1);
            if (i + 2 < n) DS41_Q8_TERM(2);
            if (i + 3 < n) DS41_Q8_TERM(3);
#undef DS41_Q8_TERM
        }
    }
    return sum;
}

static __m512 ds41_cpu_q4_rows16(const uint8_t *rows, const float *x,
                                uint64_t in_dim, uint64_t row_bytes) {
    const __m512i offsets = ds41_cpu_row_offsets(row_bytes);
    __m512 sum = _mm512_setzero_ps();
    for (uint64_t b = 0; b < in_dim / QK_K; b++) {
        const uint8_t *block = rows + b * sizeof(block_q4_K);
        const __m512i halves = _mm512_i32gather_epi32(offsets, block, 1);
        const __m512 d = ds41_cpu_half16(halves);
        const __m512 dmin = ds41_cpu_half16(_mm512_srli_epi32(halves, 16));
        for (unsigned g = 0; g < 8; g++) {
            float scales[16] __attribute__((aligned(64)));
            float mins[16] __attribute__((aligned(64)));
            for (unsigned lane = 0; lane < 16; lane++) {
                uint8_t sc, mn;
                q4_k_get_scale_min((int)g, block + lane * row_bytes + 4, &sc, &mn);
                scales[lane] = (float)sc;
                mins[lane] = (float)mn;
            }
            const __m512 scale = _mm512_mul_ps(d, _mm512_loadu_ps(scales));
            const __m512 minv = _mm512_mul_ps(dmin, _mm512_loadu_ps(mins));
            for (unsigned i = 0; i < 32; i += 4) {
                __m512i packed = _mm512_i32gather_epi32(
                    offsets, block + 16 + (g / 2) * 32 + i, 1);
                packed = _mm512_srlv_epi32(packed, _mm512_set1_epi32((g & 1) * 4));
#define DS41_Q4_TERM(J) do { \
    const __m512 q = _mm512_cvtepi32_ps(_mm512_and_si512( \
        _mm512_srli_epi32(packed, 8 * (J)), _mm512_set1_epi32(15))); \
    const __m512 value = _mm512_sub_ps(_mm512_mul_ps(scale, q), minv); \
    sum = _mm512_add_ps(sum, _mm512_mul_ps(value, \
        _mm512_set1_ps(x[b * QK_K + g * 32 + i + (J)]))); \
} while (0)
                DS41_Q4_TERM(0);
                DS41_Q4_TERM(1);
                DS41_Q4_TERM(2);
                DS41_Q4_TERM(3);
#undef DS41_Q4_TERM
            }
        }
    }
    return sum;
}

/* ds4_vec_dot_iq2_xxs_f32 per lane: (0.125 * d) * (2 * s + 1) as the scale,
 * then (scale * signed grid byte) * x added in row order. Sign bytes are
 * ksigns_iq2xs[i] = i | parity(i) << 7, computed instead of gathered. */
static __m512 ds41_cpu_iq2xxs_rows16(const uint8_t *rows, const float *x,
                                    uint64_t in_dim, uint64_t row_bytes) {
    const __m512i offsets = ds41_cpu_row_offsets(row_bytes);
    const __m512i byte = _mm512_set1_epi32(255), zero = _mm512_setzero_si512();
    const long long *grid = (const long long *)iq2xxs_grid;
    __m512 sum = _mm512_setzero_ps();
    for (uint64_t b = 0; b < in_dim / QK_K; b++) {
        const uint8_t *block = rows + b * sizeof(block_iq2_xxs);
        const __m512 d = _mm512_mul_ps(_mm512_set1_ps(0.125f),
            ds41_cpu_half16(_mm512_i32gather_epi32(offsets, block, 1)));
        for (unsigned ib = 0; ib < QK_K / 32; ib++) {
            const uint8_t *q = block + offsetof(block_iq2_xxs, qs) + ib * 8u;
            const __m512i aux0 = _mm512_i32gather_epi32(offsets, q, 1);
            const __m512i aux1 = _mm512_i32gather_epi32(offsets, q + 4, 1);
            const __m512 scale = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_add_epi32(
                _mm512_slli_epi32(_mm512_srli_epi32(aux1, 28), 1), _mm512_set1_epi32(1))));
            const float *xb = x + b * QK_K + ib * 32u;
            for (unsigned l = 0; l < 4; l++) {
                const __m512i idx = _mm512_and_si512(_mm512_srli_epi32(aux0, 8 * l), byte);
                const __m512i lo = _mm512_i32gather_epi64(_mm512_castsi512_si256(idx), grid, 8);
                const __m512i hi = _mm512_i32gather_epi64(_mm512_extracti64x4_epi64(idx, 1), grid, 8);
                const __m512i w0 = _mm512_inserti64x4(_mm512_castsi256_si512(
                    _mm512_cvtepi64_epi32(lo)), _mm512_cvtepi64_epi32(hi), 1);
                const __m512i w1 = _mm512_inserti64x4(_mm512_castsi256_si512(
                    _mm512_cvtepi64_epi32(_mm512_srli_epi64(lo, 32))),
                    _mm512_cvtepi64_epi32(_mm512_srli_epi64(hi, 32)), 1);
                const __m512i s7 = _mm512_and_si512(_mm512_srli_epi32(aux1, 7 * l),
                                                    _mm512_set1_epi32(127));
                __m512i parity = _mm512_xor_si512(s7, _mm512_srli_epi32(s7, 4));
                parity = _mm512_xor_si512(parity, _mm512_srli_epi32(parity, 2));
                parity = _mm512_xor_si512(parity, _mm512_srli_epi32(parity, 1));
                const __m512i signs = _mm512_or_si512(s7, _mm512_slli_epi32(
                    _mm512_and_si512(parity, _mm512_set1_epi32(1)), 7));
                for (unsigned j = 0; j < 8; j++) {
                    const __m512i g = _mm512_and_si512(
                        _mm512_srli_epi32(j < 4 ? w0 : w1, 8 * (j & 3u)), byte);
                    const __mmask16 neg = _mm512_test_epi32_mask(signs, _mm512_set1_epi32(1 << j));
                    const __m512 v = _mm512_cvtepi32_ps(
                        _mm512_mask_sub_epi32(g, neg, zero, g));
                    sum = _mm512_add_ps(sum, _mm512_mul_ps(_mm512_mul_ps(scale, v),
                                                           _mm512_set1_ps(xb[l * 8u + j])));
                }
            }
        }
    }
    return sum;
}

/* q2_k_value_f32 per lane: (d * (sc & 15)) * q - dmin * (sc >> 4), then
 * one product and one addition per element in row order. */
static __m512 ds41_cpu_q2k_rows16(const uint8_t *rows, const float *x,
                                 uint64_t in_dim, uint64_t row_bytes) {
    const __m512i offsets = ds41_cpu_row_offsets(row_bytes);
    const __m512i low4 = _mm512_set1_epi32(15), low2 = _mm512_set1_epi32(3);
    __m512 sum = _mm512_setzero_ps();
    for (uint64_t b = 0; b < in_dim / QK_K; b++) {
        const uint8_t *block = rows + b * sizeof(block_q2_K);
        const __m512i halves = _mm512_i32gather_epi32(
            offsets, block + offsetof(block_q2_K, d), 1);
        const __m512 d = ds41_cpu_half16(halves);
        const __m512 dmin = ds41_cpu_half16(_mm512_srli_epi32(halves, 16));
        for (unsigned g = 0; g < QK_K / 16; g++) {
            const __m512i sc = _mm512_and_si512(_mm512_srli_epi32(
                _mm512_i32gather_epi32(offsets, block + (g & ~3u), 1),
                8 * (g & 3u)), _mm512_set1_epi32(255));
            const __m512 scale = _mm512_mul_ps(d, _mm512_cvtepi32_ps(
                _mm512_and_si512(sc, low4)));
            const __m512 minv = _mm512_mul_ps(dmin, _mm512_cvtepi32_ps(
                _mm512_srli_epi32(sc, 4)));
            const unsigned q_base = 32u * (g / 8u) + 16u * (g & 1u);
            const __m512i shift = _mm512_set1_epi32((int)(((g / 2u) & 3u) * 2u));
            const float *xg = x + b * QK_K + g * 16u;
            for (unsigned l = 0; l < 16; l += 4) {
                const __m512i packed = _mm512_srlv_epi32(_mm512_i32gather_epi32(
                    offsets, block + offsetof(block_q2_K, qs) + q_base + l, 1), shift);
#define DS41_Q2K_TERM(J) do { \
    const __m512 q = _mm512_cvtepi32_ps(_mm512_and_si512( \
        _mm512_srli_epi32(packed, 8 * (J)), low2)); \
    const __m512 value = _mm512_sub_ps(_mm512_mul_ps(scale, q), minv); \
    sum = _mm512_add_ps(sum, _mm512_mul_ps(value, _mm512_set1_ps(xg[l + (J)]))); \
} while (0)
                DS41_Q2K_TERM(0);
                DS41_Q2K_TERM(1);
                DS41_Q2K_TERM(2);
                DS41_Q2K_TERM(3);
#undef DS41_Q2K_TERM
            }
        }
    }
    return sum;
}

/* Rows [g * group_rows, (g + 1) * group_rows) read ref.x + g * group_in. */
typedef struct {
    matvec_q8_0_f32_ref_ctx ref;
    uint64_t rows, group_rows, group_in;
} ds41_cpu_q8_ctx;

static void ds41_cpu_q8_worker(void *vctx, uint64_t first, uint64_t last) {
    ds41_cpu_q8_ctx *ctx = vctx;
    const uint64_t row_bytes = ctx->ref.blocks * 34;
    for (uint64_t tile = first; tile < last; tile++) {
        const uint64_t row = tile * 16;
        const float *x = ctx->ref.x + row / ctx->group_rows * ctx->group_in;
        if (ctx->rows - row >= 16) {
            _mm512_storeu_ps(ctx->ref.out + row, ds41_cpu_q8_rows16(
                ctx->ref.data + row * row_bytes, x, ctx->ref.in_dim, row_bytes));
        } else {
            matvec_q8_0_f32_ref_ctx ref = ctx->ref;
            ref.x = x;
            matvec_q8_0_f32_ref_worker(&ref, row, ctx->rows);
        }
    }
}

/* Units [0, units) are 16-row tiles of m[0], the next units those of m[1]. */
typedef struct {
    ds41_cpu_q8_ctx m[2];
    uint64_t units;
} ds41_cpu_q8_pair_ctx;

static void ds41_cpu_q8_pair_worker(void *vctx, uint64_t first, uint64_t last) {
    ds41_cpu_q8_pair_ctx *ctx = vctx;
    while (first < last) {
        const unsigned k = first >= ctx->units;
        const uint64_t base = k ? ctx->units : 0;
        const uint64_t end = last < base + ctx->units ? last : base + ctx->units;
        ds41_cpu_q8_worker(&ctx->m[k], first - base, end - base);
        first = end;
    }
}

static void ds41_cpu_mid_worker(void *vctx, uint64_t first, uint64_t last) {
    glm_routed_moe_f32_mid_ctx *ctx = vctx;
    const uint64_t tiles = (ctx->out_dim + 15) / 16;
    for (uint64_t tile = first; tile < last; tile++) {
        const uint32_t slot = (uint32_t)(tile / tiles);
        const uint64_t row = (tile % tiles) * 16;
        const uint64_t idx = (uint64_t)slot * ctx->out_dim + row;
        if (ctx->out_dim - row < 16) {
            glm_routed_moe_f32_mid_worker(ctx, idx, (uint64_t)(slot + 1) * ctx->out_dim);
            continue;
        }
        float gate[16] __attribute__((aligned(64)));
        float up[16] __attribute__((aligned(64)));
        __m512 (*rows16)(const uint8_t *, const float *, uint64_t, uint64_t) =
            ctx->gate_type == DS4_TENSOR_IQ2_XXS ? ds41_cpu_iq2xxs_rows16 : ds41_cpu_q4_rows16;
        _mm512_storeu_ps(gate, rows16(
            ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot], ctx->x,
            ctx->in_dim, ctx->gate_row_bytes[slot]));
        _mm512_storeu_ps(up, rows16(
            ctx->up_base[slot] + row * ctx->up_row_bytes[slot], ctx->x,
            ctx->in_dim, ctx->up_row_bytes[slot]));
        for (unsigned lane = 0; lane < 16; lane++) {
            float g = gate[lane], u = up[lane];
            if (ctx->clamp > 1.0e-6f) {
                if (g > ctx->clamp) g = ctx->clamp;
                if (u > ctx->clamp) u = ctx->clamp;
                if (u < -ctx->clamp) u = -ctx->clamp;
            }
            ctx->mid[idx + lane] = silu(g) * u * ctx->expert_weight[slot];
        }
    }
}

static void ds41_cpu_down_worker(void *vctx, uint64_t first, uint64_t last) {
    glm_routed_moe_f32_down_ctx *ctx = vctx;
    for (uint64_t tile = first; tile < last; tile++) {
        const uint64_t row = tile * 16;
        if (ctx->out_dim - row < 16) {
            glm_routed_moe_f32_down_worker(ctx, row, ctx->out_dim);
            continue;
        }
        __m512 sum = _mm512_setzero_ps();
        for (uint32_t slot = 0; slot < ctx->n_expert; slot++) {
            const uint8_t *rows = ctx->down_base[slot] + row * ctx->down_row_bytes[slot];
            const float *mid = ctx->mid + (uint64_t)slot * ctx->in_dim;
            sum = _mm512_add_ps(sum, ctx->down_type == DS4_TENSOR_Q2_K ?
                ds41_cpu_q2k_rows16(rows, mid, ctx->in_dim, ctx->down_row_bytes[slot]) :
                ds41_cpu_q4_rows16(rows, mid, ctx->in_dim, ctx->down_row_bytes[slot]));
        }
        _mm512_storeu_ps(ctx->out + row, sum);
    }
}
#endif

/* A matrix whose row groups read consecutive input slices, like the grouped
 * output projection, in one pool dispatch instead of one per group: rows
 * [g * group_rows, (g + 1) * group_rows) read x + g * group_in.  group_rows
 * must divide the rows; a multiple of 16 keeps the tiles inside one group. */
static void ds41_cpu_matvec_q8_grouped(float *out, const ds4_model *m,
                                      const ds4_tensor *t, const float *x,
                                      uint64_t group_rows, uint64_t group_in) {
    if (t->type != DS4_TENSOR_Q8_0 || t->ndim < 2 || t->dim[0] == 0)
        ds4_die("expected a Q8_0 tensor with matrix rows");
    const uint64_t rows = t->elements / t->dim[0];
    const uint64_t blocks = (t->dim[0] + 31) / 32;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (t->dim[0] <= (uint64_t)(INT32_MAX / (15 * 34)) * 32 &&
        (group_rows == rows || group_rows % 16 == 0)) {
        ds41_cpu_q8_ctx ctx = {
            .ref = {.out = out, .data = tensor_data(m, t), .x = x,
                    .in_dim = t->dim[0], .blocks = blocks},
            .rows = rows, .group_rows = group_rows, .group_in = group_in,
        };
        /* The old threshold was 512 rows; pool work units are now 16 rows. */
        ds4_parallel_for_min_rows((ctx.rows + 15) / 16, ds41_cpu_q8_worker, &ctx, 32);
        return;
    }
#endif
    for (uint64_t row = 0; row < rows; row += group_rows) {
        matvec_q8_0_f32_ref_ctx ctx = {
            .out = out + row, .data = (const uint8_t *)tensor_data(m, t) + row * blocks * 34,
            .x = x + row / group_rows * group_in, .in_dim = t->dim[0], .blocks = blocks,
        };
        ds4_parallel_for(group_rows, matvec_q8_0_f32_ref_worker, &ctx);
    }
}

static void ds41_cpu_matvec_q8(float *out, const ds4_model *m,
                              const ds4_tensor *t, const float *x) {
    ds41_cpu_matvec_q8_grouped(out, m, t, x, t->elements / t->dim[0], 0);
}

/* Two Q8_0 matrices of the same shape and input, such as the shared expert's
 * gate and up projections, in one dispatch.  Each matrix is its own NUMA
 * group, so a node takes the half of both that the warm pass placed on it.
 * Returns false, having done nothing, when the pair does not qualify. */
static bool ds41_cpu_matvec_q8_pair(float *out0, float *out1, const ds4_model *m,
                                   const ds4_tensor *t0, const ds4_tensor *t1,
                                   const float *x) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (t0->type != DS4_TENSOR_Q8_0 || t1->type != DS4_TENSOR_Q8_0 ||
        t0->ndim < 2 || t1->ndim < 2 || t0->dim[0] == 0 || t0->dim[0] != t1->dim[0] ||
        t0->elements != t1->elements || t0->elements < t0->dim[0] ||
        t0->dim[0] > (uint64_t)(INT32_MAX / (15 * 34)) * 32) return false;
    const uint64_t rows = t0->elements / t0->dim[0];
    ds41_cpu_q8_pair_ctx ctx = {.units = (rows + 15) / 16};
    for (unsigned k = 0; k < 2; k++) {
        ctx.m[k] = (ds41_cpu_q8_ctx){
            .ref = {.out = k ? out1 : out0, .data = tensor_data(m, k ? t1 : t0), .x = x,
                    .in_dim = t0->dim[0], .blocks = (t0->dim[0] + 31) / 32},
            .rows = rows, .group_rows = rows, .group_in = 0,
        };
    }
    ds4_pool_dispatch(2 * ctx.units, ds41_cpu_q8_pair_worker, &ctx, ctx.units, 32);
    return true;
#else
    (void)out0; (void)out1; (void)m; (void)t0; (void)t1; (void)x;
    return false;
#endif
}

static void ds41_cpu_routed_moe(float *out, float *mid, const ds4_model *m,
                               const ds4_layer_weights *l, const float *x,
                               const int *selected, const float *weights) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const bool q4 = l->ffn_gate_exps && l->ffn_up_exps && l->ffn_down_exps &&
        l->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_down_exps->type == DS4_TENSOR_Q4_K;
    /* Q2 models: IQ2_XXS gate/up and Q2_K down, also one row per lane. */
    const bool q2 = l->ffn_gate_exps && l->ffn_up_exps && l->ffn_down_exps &&
        l->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        l->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        l->ffn_down_exps->type == DS4_TENSOR_Q2_K;
    if ((q4 || q2) &&
        DS4_N_EXPERT_USED > 0 && DS4_N_EXPERT_USED <= DS4_MAX_EXPERT_USED &&
        DS4_N_EMBD > 0 && DS4_N_FF_EXP > 0 &&
        DS4_N_EMBD % QK_K == 0 && DS4_N_FF_EXP % QK_K == 0 &&
        DS4_N_EMBD / QK_K <= INT32_MAX / (15 * (int)sizeof(block_q4_K)) &&
        DS4_N_FF_EXP / QK_K <= INT32_MAX / (15 * (int)sizeof(block_q4_K))) {
        glm_routed_moe_f32_mid_ctx mc = {
            .mid = mid, .x = x, .in_dim = DS4_N_EMBD, .out_dim = DS4_N_FF_EXP,
            .gate_type = l->ffn_gate_exps->type, .up_type = l->ffn_up_exps->type,
            .n_expert = DS4_N_EXPERT_USED, .clamp = DS4_SWIGLU_CLAMP_EXP,
        };
        glm_routed_moe_f32_down_ctx dc = {
            .out = out, .mid = mid, .in_dim = DS4_N_FF_EXP, .out_dim = DS4_N_EMBD,
            .down_type = l->ffn_down_exps->type, .n_expert = DS4_N_EXPERT_USED,
        };
        for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
            uint64_t gi, go, ui, uo, di, dout;
            if (selected[slot] < 0 || (uint32_t)selected[slot] >= DS4_N_EXPERT)
                ds4_die("V4.1 CPU routed-MoE selected expert is outside range");
            mc.gate_base[slot] = tensor_expert_bytes(m, l->ffn_gate_exps,
                (uint32_t)selected[slot], &gi, &go, &mc.gate_row_bytes[slot]);
            mc.up_base[slot] = tensor_expert_bytes(m, l->ffn_up_exps,
                (uint32_t)selected[slot], &ui, &uo, &mc.up_row_bytes[slot]);
            dc.down_base[slot] = tensor_expert_bytes(m, l->ffn_down_exps,
                (uint32_t)selected[slot], &di, &dout, &dc.down_row_bytes[slot]);
            if (gi != mc.in_dim || ui != mc.in_dim || go != mc.out_dim ||
                uo != mc.out_dim || di != dc.in_dim || dout != dc.out_dim)
                ds4_die("V4.1 CPU routed-MoE found unexpected expert dimensions");
            mc.expert_weight[slot] = weights[slot];
        }
        /* Every expert is its own NUMA group, so each node takes the first
         * or second half of every selected expert's gate and up rows, the
         * halves --numa-place all put on it, instead of three whole experts
         * read half from the other socket. */
        const uint64_t tiles = (mc.out_dim + 15) / 16;
        ds4_pool_dispatch((uint64_t)mc.n_expert * tiles, ds41_cpu_mid_worker, &mc, tiles, 32);
        ds4_parallel_for_min_rows((dc.out_dim + 15) / 16, ds41_cpu_down_worker, &dc, 32);
        return;
    }
#endif
    layer_glm_routed_moe_one_f32_ref(out, mid, m, l, x, selected, weights);
}

#if defined(__clang__)
#pragma float_control(pop)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif
