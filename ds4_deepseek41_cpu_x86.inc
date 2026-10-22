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

/* IQ2_XXS grid entries as 2-bit magnitude codes (8 -> 0, 25 -> 1, 43 -> 2),
 * element j at bits 2 * j, for in-register VPERMI2W lookups. */
static uint16_t ds41_iq2xxs_codes[256] __attribute__((aligned(64)));
static pthread_once_t ds41_iq2xxs_once = PTHREAD_ONCE_INIT;

static void ds41_iq2xxs_tables_init(void) {
    for (uint32_t g = 0; g < 256; g++) {
        const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + g);
        uint32_t codes = 0;
        for (uint32_t j = 0; j < 8; j++) {
            if (grid[j] != 8 && grid[j] != 25 && grid[j] != 43)
                ds4_die("unexpected IQ2_XXS grid value");
            codes |= (uint32_t)((grid[j] > 8) + (grid[j] > 25)) << (2 * j);
        }
        ds41_iq2xxs_codes[g] = (uint16_t)codes;
    }
}

/* The codes of the four grid indices in each dword lane of idx: lane r of
 * c01 holds those of bytes 0 and 1 in its two words, lane r of c23 those of
 * bytes 2 and 3.  Four 64-entry VPERMI2W lookups are chosen by index bits 6
 * and 7. */
static inline void ds41_iq2xxs_codes16(__m512i idx, const __m512i *t,
                                      __m512i *c01, __m512i *c23) {
    __m512i c[2];
    for (unsigned h = 0; h < 2; h++) {
        const __m512i w = _mm512_cvtepu8_epi16(h ? _mm512_extracti64x4_epi64(idx, 1) :
                                                   _mm512_castsi512_si256(idx));
        const __mmask32 bit6 = _mm512_test_epi16_mask(w, _mm512_set1_epi16(64));
        const __mmask32 bit7 = _mm512_test_epi16_mask(w, _mm512_set1_epi16(128));
        c[h] = _mm512_mask_blend_epi16(bit7,
            _mm512_mask_blend_epi16(bit6, _mm512_permutex2var_epi16(t[0], w, t[1]),
                                          _mm512_permutex2var_epi16(t[2], w, t[3])),
            _mm512_mask_blend_epi16(bit6, _mm512_permutex2var_epi16(t[4], w, t[5]),
                                          _mm512_permutex2var_epi16(t[6], w, t[7])));
    }
    *c01 = _mm512_permutex2var_epi32(c[0], _mm512_setr_epi32(
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30), c[1]);
    *c23 = _mm512_permutex2var_epi32(c[0], _mm512_setr_epi32(
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31), c[1]);
}

/* Bit 7 * l is the parity of the 7-bit sign index at bits 7 * l .. 7 * l + 6,
 * the last bit of ksigns_iq2xs[index]. */
static inline __m512i ds41_iq2xxs_parity(__m512i aux) {
    const __m512i p1 = _mm512_xor_si512(aux, _mm512_srli_epi32(aux, 1));
    const __m512i p2 = _mm512_xor_si512(p1, _mm512_srli_epi32(p1, 2));
    return _mm512_ternarylogic_epi32(p2, _mm512_srli_epi32(p1, 4),
                                     _mm512_srli_epi32(aux, 6), 0x96);
}

/* ds4_vec_dot_iq2_xxs_f32 per lane for 16 rows of two matrices at once, such
 * as an expert's gate and up rows: two independent sums that share the
 * activation broadcasts.  Each lane computes (0.125 * d) * (2 * s + 1) as the
 * scale and adds (scale * signed grid value) * x in row order, the reference's
 * operations in its order.  The grid values are built in registers, a VPERMPS
 * of the magnitude code with the ksigns_iq2xs bit xored into the sign, so the
 * only gathers left read the 16 rows' blocks.  ds41_iq2xxs_codes must be
 * initialized. */
static void ds41_cpu_iq2xxs_pair16(__m512 out[2], const uint8_t *rows0, uint64_t row_bytes0,
                                  const uint8_t *rows1, uint64_t row_bytes1,
                                  const float *x, uint64_t in_dim) {
    const __m512i off0 = ds41_cpu_row_offsets(row_bytes0);
    const __m512i off1 = ds41_cpu_row_offsets(row_bytes1);
    const __m512i one = _mm512_set1_epi32(1), sign = _mm512_set1_epi32((int)0x80000000u);
    const __m512 values = _mm512_setr_ps(8.0f, 25.0f, 43.0f, 0.0f, 8.0f, 25.0f, 43.0f, 0.0f,
                                         8.0f, 25.0f, 43.0f, 0.0f, 8.0f, 25.0f, 43.0f, 0.0f);
    const __m512i *t = (const __m512i *)ds41_iq2xxs_codes;
    __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
    for (uint64_t b = 0; b < in_dim / QK_K; b++) {
        const uint8_t *block0 = rows0 + b * sizeof(block_iq2_xxs);
        const uint8_t *block1 = rows1 + b * sizeof(block_iq2_xxs);
        const __m512 d0 = _mm512_mul_ps(_mm512_set1_ps(0.125f),
            ds41_cpu_half16(_mm512_i32gather_epi32(off0, block0, 1)));
        const __m512 d1 = _mm512_mul_ps(_mm512_set1_ps(0.125f),
            ds41_cpu_half16(_mm512_i32gather_epi32(off1, block1, 1)));
        for (unsigned ib = 0; ib < QK_K / 32; ib++) {
            const uint8_t *q0 = block0 + offsetof(block_iq2_xxs, qs) + ib * 8u;
            const uint8_t *q1 = block1 + offsetof(block_iq2_xxs, qs) + ib * 8u;
            const __m512i aux0 = _mm512_i32gather_epi32(off0, q0 + 4, 1);
            const __m512i aux1 = _mm512_i32gather_epi32(off1, q1 + 4, 1);
            __m512i c0[4], c1[4];
            ds41_iq2xxs_codes16(_mm512_i32gather_epi32(off0, q0, 1), t, &c0[0], &c0[2]);
            ds41_iq2xxs_codes16(_mm512_i32gather_epi32(off1, q1, 1), t, &c1[0], &c1[2]);
            c0[1] = _mm512_srli_epi32(c0[0], 16); c0[3] = _mm512_srli_epi32(c0[2], 16);
            c1[1] = _mm512_srli_epi32(c1[0], 16); c1[3] = _mm512_srli_epi32(c1[2], 16);
            const __m512i par0 = ds41_iq2xxs_parity(aux0), par1 = ds41_iq2xxs_parity(aux1);
            const __m512 scale0 = _mm512_mul_ps(d0, _mm512_cvtepi32_ps(_mm512_add_epi32(
                _mm512_slli_epi32(_mm512_srli_epi32(aux0, 28), 1), one)));
            const __m512 scale1 = _mm512_mul_ps(d1, _mm512_cvtepi32_ps(_mm512_add_epi32(
                _mm512_slli_epi32(_mm512_srli_epi32(aux1, 28), 1), one)));
            for (unsigned l = 0; l < 4; l++) {
                /* Group l: codes at bits 2 * j, sign index at bits 0-6 and
                 * its parity at bit 0 after shifting by 7 * l. */
                const __m512i shift = _mm512_set1_epi32((int)(7 * l));
                const __m512i code0 = c0[l], code1 = c1[l];
                const __m512i s0 = _mm512_srlv_epi32(aux0, shift), s1 = _mm512_srlv_epi32(aux1, shift);
                const __m512i p0 = _mm512_srlv_epi32(par0, shift), p1 = _mm512_srlv_epi32(par1, shift);
                const float *xl = x + b * QK_K + ib * 32u + l * 8u;
/* Element J: magnitude by VPERMPS of its code, then the sign bit of
 * ksigns_iq2xs[index] (index bit J, or the parity for J = 7) xored in. */
#define DS41_IQ2_VALUE(CODE, S, P, J) _mm512_castsi512_ps(_mm512_ternarylogic_epi32( \
    _mm512_castps_si512(_mm512_permutexvar_ps(_mm512_srli_epi32(CODE, 2 * (J)), values)), \
    (J) < 7 ? _mm512_slli_epi32(S, 31 - (J)) : _mm512_slli_epi32(P, 31), sign, 0x78))
#define DS41_IQ2_TERM(J) do { \
    const __m512 xj = _mm512_set1_ps(xl[J]); \
    const __m512 v0 = DS41_IQ2_VALUE(code0, s0, p0, J); \
    const __m512 v1 = DS41_IQ2_VALUE(code1, s1, p1, J); \
    sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(_mm512_mul_ps(scale0, v0), xj)); \
    sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(_mm512_mul_ps(scale1, v1), xj)); \
} while (0)
                DS41_IQ2_TERM(0);
                DS41_IQ2_TERM(1);
                DS41_IQ2_TERM(2);
                DS41_IQ2_TERM(3);
                DS41_IQ2_TERM(4);
                DS41_IQ2_TERM(5);
                DS41_IQ2_TERM(6);
                DS41_IQ2_TERM(7);
            }
#undef DS41_IQ2_TERM
#undef DS41_IQ2_VALUE
        }
    }
    out[0] = sum0;
    out[1] = sum1;
}

/* q2_k_value_f32 per lane: (d * (sc & 15)) * q - dmin * (sc >> 4), then one
 * product and one addition per element in row order, for 16 rows of two
 * matrices with their own inputs, such as two experts' down rows: two
 * independent sums.  The 32 quant bytes of half a block hold the 2-bit values
 * of its eight groups at four shifts, so they are gathered once into eight
 * registers rather than once per group, and a scale dword serves four groups. */
static void ds41_cpu_q2k_pair16(__m512 out[2],
                               const uint8_t *rows0, uint64_t row_bytes0, const float *x0,
                               const uint8_t *rows1, uint64_t row_bytes1, const float *x1,
                               uint64_t in_dim) {
    const __m512i off0 = ds41_cpu_row_offsets(row_bytes0);
    const __m512i off1 = ds41_cpu_row_offsets(row_bytes1);
    const __m512i low4 = _mm512_set1_epi32(15), byte = _mm512_set1_epi32(255);
    /* (float)q by VPERMPS of the 2-bit value in index bits 0-1. */
    const __m512 qf = _mm512_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f,
                                     0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f);
    __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
    for (uint64_t b = 0; b < in_dim / QK_K; b++) {
        const uint8_t *block0 = rows0 + b * sizeof(block_q2_K);
        const uint8_t *block1 = rows1 + b * sizeof(block_q2_K);
        const __m512i halves0 = _mm512_i32gather_epi32(off0, block0 + offsetof(block_q2_K, d), 1);
        const __m512i halves1 = _mm512_i32gather_epi32(off1, block1 + offsetof(block_q2_K, d), 1);
        const __m512 d0 = ds41_cpu_half16(halves0), dmin0 = ds41_cpu_half16(_mm512_srli_epi32(halves0, 16));
        const __m512 d1 = ds41_cpu_half16(halves1), dmin1 = ds41_cpu_half16(_mm512_srli_epi32(halves1, 16));
        for (unsigned c = 0; c < 2; c++) {
            __m512i q0[8], q1[8];
            for (unsigned k = 0; k < 8; k++) {
                q0[k] = _mm512_i32gather_epi32(off0, block0 + offsetof(block_q2_K, qs) + 32u * c + 4u * k, 1);
                q1[k] = _mm512_i32gather_epi32(off1, block1 + offsetof(block_q2_K, qs) + 32u * c + 4u * k, 1);
            }
            __m512i sc_word0 = _mm512_setzero_si512(), sc_word1 = _mm512_setzero_si512();
            for (unsigned gq = 0; gq < 8; gq++) {
                const unsigned g = 8u * c + gq;
                if ((gq & 3u) == 0) {
                    sc_word0 = _mm512_i32gather_epi32(off0, block0 + g, 1);
                    sc_word1 = _mm512_i32gather_epi32(off1, block1 + g, 1);
                }
                const __m512i sc_shift = _mm512_set1_epi32((int)(8u * (gq & 3u)));
                const __m512i sc0 = _mm512_and_si512(_mm512_srlv_epi32(sc_word0, sc_shift), byte);
                const __m512i sc1 = _mm512_and_si512(_mm512_srlv_epi32(sc_word1, sc_shift), byte);
                const __m512 scale0 = _mm512_mul_ps(d0, _mm512_cvtepi32_ps(_mm512_and_si512(sc0, low4)));
                const __m512 minv0 = _mm512_mul_ps(dmin0, _mm512_cvtepi32_ps(_mm512_srli_epi32(sc0, 4)));
                const __m512 scale1 = _mm512_mul_ps(d1, _mm512_cvtepi32_ps(_mm512_and_si512(sc1, low4)));
                const __m512 minv1 = _mm512_mul_ps(dmin1, _mm512_cvtepi32_ps(_mm512_srli_epi32(sc1, 4)));
                const __m512i shift = _mm512_set1_epi32((int)(((g / 2u) & 3u) * 2u));
                const __m512i *qg0 = q0 + 4u * (g & 1u), *qg1 = q1 + 4u * (g & 1u);
                const float *xg0 = x0 + b * QK_K + g * 16u, *xg1 = x1 + b * QK_K + g * 16u;
                for (unsigned l = 0; l < 16; l += 4) {
                    const __m512i packed0 = _mm512_srlv_epi32(qg0[l / 4u], shift);
                    const __m512i packed1 = _mm512_srlv_epi32(qg1[l / 4u], shift);
#define DS41_Q2K_TERM(J) do { \
    const __m512 v0 = _mm512_sub_ps(_mm512_mul_ps(scale0, _mm512_permutexvar_ps( \
        (J) ? _mm512_srli_epi32(packed0, 8 * (J)) : packed0, qf)), minv0); \
    const __m512 v1 = _mm512_sub_ps(_mm512_mul_ps(scale1, _mm512_permutexvar_ps( \
        (J) ? _mm512_srli_epi32(packed1, 8 * (J)) : packed1, qf)), minv1); \
    sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(v0, _mm512_set1_ps(xg0[l + (J)]))); \
    sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(v1, _mm512_set1_ps(xg1[l + (J)]))); \
} while (0)
                    DS41_Q2K_TERM(0);
                    DS41_Q2K_TERM(1);
                    DS41_Q2K_TERM(2);
                    DS41_Q2K_TERM(3);
#undef DS41_Q2K_TERM
                }
            }
        }
    }
    out[0] = sum0;
    out[1] = sum1;
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
        const uint8_t *gate_rows = ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot];
        const uint8_t *up_rows = ctx->up_base[slot] + row * ctx->up_row_bytes[slot];
        if (ctx->gate_type == DS4_TENSOR_IQ2_XXS) {
            __m512 sums[2];
            ds41_cpu_iq2xxs_pair16(sums, gate_rows, ctx->gate_row_bytes[slot],
                                   up_rows, ctx->up_row_bytes[slot], ctx->x, ctx->in_dim);
            _mm512_storeu_ps(gate, sums[0]);
            _mm512_storeu_ps(up, sums[1]);
        } else {
            _mm512_storeu_ps(gate, ds41_cpu_q4_rows16(gate_rows, ctx->x, ctx->in_dim,
                                                      ctx->gate_row_bytes[slot]));
            _mm512_storeu_ps(up, ds41_cpu_q4_rows16(up_rows, ctx->x, ctx->in_dim,
                                                    ctx->up_row_bytes[slot]));
        }
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
        /* Experts are added in slot order; Q2_K experts are computed in pairs. */
        __m512 sum = _mm512_setzero_ps();
        for (uint32_t slot = 0; slot < ctx->n_expert; ) {
            const uint8_t *rows = ctx->down_base[slot] + row * ctx->down_row_bytes[slot];
            const float *mid = ctx->mid + (uint64_t)slot * ctx->in_dim;
            if (ctx->down_type == DS4_TENSOR_Q2_K) {
                const uint32_t next = slot + 1 < ctx->n_expert ? slot + 1 : slot;
                __m512 sums[2];
                ds41_cpu_q2k_pair16(sums, rows, ctx->down_row_bytes[slot], mid,
                                    ctx->down_base[next] + row * ctx->down_row_bytes[next],
                                    ctx->down_row_bytes[next],
                                    ctx->mid + (uint64_t)next * ctx->in_dim, ctx->in_dim);
                sum = _mm512_add_ps(sum, sums[0]);
                if (next != slot) sum = _mm512_add_ps(sum, sums[1]);
                slot = next + 1;
            } else {
                sum = _mm512_add_ps(sum, ds41_cpu_q4_rows16(rows, mid, ctx->in_dim,
                                                            ctx->down_row_bytes[slot]));
                slot++;
            }
        }
        _mm512_storeu_ps(ctx->out + row, sum);
    }
}

/* Batched prefill kernels.  A tile's weights are decoded once per block and
 * applied to many tokens.  For every (row, token) the products and additions
 * are those of the one-token kernels above, in the same order, so each
 * output is bitwise the decode result.  Tokens advance in pairs or quads
 * only to give the core independent addition chains. */
#define DS41_CPU_PASS 32u

static inline __m512 ds41_cpu_bf16_16(__m512 v) {
    const __m512i bits = _mm512_castps_si512(v);
    const __m512i exp = _mm512_set1_epi32(0x7f800000);
    const __mmask16 finite = _mm512_cmpneq_epi32_mask(_mm512_and_si512(bits, exp), exp);
    const __m512i bias = _mm512_add_epi32(_mm512_set1_epi32(0x7fff), _mm512_and_si512(
        _mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1)));
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_mask_add_epi32(bits, finite, bits, bias),
                                                _mm512_set1_epi32((int)0xffff0000u)));
}

/* acc[t] += w[i] * x[t][off + i] for i in [0, n), in order of i. */
static inline void ds41_cpu_apply16(__m512 *acc, const __m512 *w, const float *const *x,
                                    uint32_t nt, uint64_t off, uint32_t n) {
    uint32_t t = 0;
    for (; t + 4 <= nt; t += 4) {
        const float *x0 = x[t] + off, *x1 = x[t + 1] + off;
        const float *x2 = x[t + 2] + off, *x3 = x[t + 3] + off;
        __m512 a0 = acc[t], a1 = acc[t + 1], a2 = acc[t + 2], a3 = acc[t + 3];
        for (uint32_t i = 0; i < n; i++) {
            a0 = _mm512_add_ps(a0, _mm512_mul_ps(w[i], _mm512_set1_ps(x0[i])));
            a1 = _mm512_add_ps(a1, _mm512_mul_ps(w[i], _mm512_set1_ps(x1[i])));
            a2 = _mm512_add_ps(a2, _mm512_mul_ps(w[i], _mm512_set1_ps(x2[i])));
            a3 = _mm512_add_ps(a3, _mm512_mul_ps(w[i], _mm512_set1_ps(x3[i])));
        }
        acc[t] = a0; acc[t + 1] = a1; acc[t + 2] = a2; acc[t + 3] = a3;
    }
    for (; t < nt; t++) {
        const float *x0 = x[t] + off;
        __m512 a0 = acc[t];
        for (uint32_t i = 0; i < n; i++)
            a0 = _mm512_add_ps(a0, _mm512_mul_ps(w[i], _mm512_set1_ps(x0[i])));
        acc[t] = a0;
    }
}

/* The same for two matrices that read the same inputs (gate and up). */
static inline void ds41_cpu_apply16x2(__m512 *a, __m512 *b, const __m512 *wa,
                                      const __m512 *wb, const float *const *x,
                                      uint32_t nt, uint64_t off, uint32_t n) {
    uint32_t t = 0;
    for (; t + 2 <= nt; t += 2) {
        const float *x0 = x[t] + off, *x1 = x[t + 1] + off;
        __m512 a0 = a[t], b0 = b[t], a1 = a[t + 1], b1 = b[t + 1];
        for (uint32_t i = 0; i < n; i++) {
            const __m512 v0 = _mm512_set1_ps(x0[i]), v1 = _mm512_set1_ps(x1[i]);
            a0 = _mm512_add_ps(a0, _mm512_mul_ps(wa[i], v0));
            b0 = _mm512_add_ps(b0, _mm512_mul_ps(wb[i], v0));
            a1 = _mm512_add_ps(a1, _mm512_mul_ps(wa[i], v1));
            b1 = _mm512_add_ps(b1, _mm512_mul_ps(wb[i], v1));
        }
        a[t] = a0; b[t] = b0; a[t + 1] = a1; b[t + 1] = b1;
    }
    if (t < nt) {
        const float *x0 = x[t] + off;
        __m512 a0 = a[t], b0 = b[t];
        for (uint32_t i = 0; i < n; i++) {
            const __m512 v0 = _mm512_set1_ps(x0[i]);
            a0 = _mm512_add_ps(a0, _mm512_mul_ps(wa[i], v0));
            b0 = _mm512_add_ps(b0, _mm512_mul_ps(wb[i], v0));
        }
        a[t] = a0; b[t] = b0;
    }
}

/* The 32 values d * q of a Q8_0 block of 16 rows, as ds41_cpu_q8_rows16. */
static inline void ds41_cpu_q8_decode16(__m512 *w, const uint8_t *block, __m512i offsets) {
    const __m512 d = ds41_cpu_half16(_mm512_i32gather_epi32(offsets, block, 1));
    for (unsigned i = 0; i < 32; i += 4) {
        const __m512i packed = _mm512_i32gather_epi32(offsets, block + 2 + i, 1);
        w[i] = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_slli_epi32(packed, 24), 24)));
        w[i + 1] = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_slli_epi32(packed, 16), 24)));
        w[i + 2] = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_slli_epi32(packed, 8), 24)));
        w[i + 3] = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_srai_epi32(packed, 24)));
    }
}

/* The 32 values scale * signed grid byte of IQ2_XXS sub-block ib, as
 * ds41_cpu_iq2xxs_rows16; d is 0.125 times the block scale. */
static inline void ds41_cpu_iq2xxs_decode16(__m512 *w, const uint8_t *block, unsigned ib,
                                            __m512 d, __m512i offsets) {
    const __m512i byte = _mm512_set1_epi32(255), zero = _mm512_setzero_si512();
    const long long *grid = (const long long *)iq2xxs_grid;
    const uint8_t *q = block + offsetof(block_iq2_xxs, qs) + ib * 8u;
    const __m512i aux0 = _mm512_i32gather_epi32(offsets, q, 1);
    const __m512i aux1 = _mm512_i32gather_epi32(offsets, q + 4, 1);
    const __m512 scale = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_add_epi32(
        _mm512_slli_epi32(_mm512_srli_epi32(aux1, 28), 1), _mm512_set1_epi32(1))));
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
            w[l * 8u + j] = _mm512_mul_ps(scale, _mm512_cvtepi32_ps(
                _mm512_mask_sub_epi32(g, neg, zero, g)));
        }
    }
}

/* The 16 values (d * (sc & 15)) * q - dmin * (sc >> 4) of Q2_K group g. */
static inline void ds41_cpu_q2k_decode16(__m512 *w, const uint8_t *block, unsigned g,
                                         __m512 d, __m512 dmin, __m512i offsets) {
    const __m512i low4 = _mm512_set1_epi32(15), low2 = _mm512_set1_epi32(3);
    const __m512i sc = _mm512_and_si512(_mm512_srli_epi32(
        _mm512_i32gather_epi32(offsets, block + (g & ~3u), 1),
        8 * (g & 3u)), _mm512_set1_epi32(255));
    const __m512 scale = _mm512_mul_ps(d, _mm512_cvtepi32_ps(_mm512_and_si512(sc, low4)));
    const __m512 minv = _mm512_mul_ps(dmin, _mm512_cvtepi32_ps(_mm512_srli_epi32(sc, 4)));
    const unsigned q_base = 32u * (g / 8u) + 16u * (g & 1u);
    const __m512i shift = _mm512_set1_epi32((int)(((g / 2u) & 3u) * 2u));
    for (unsigned l = 0; l < 16; l += 4) {
        const __m512i packed = _mm512_srlv_epi32(_mm512_i32gather_epi32(
            offsets, block + offsetof(block_q2_K, qs) + q_base + l, 1), shift);
        for (unsigned j = 0; j < 4; j++)
            w[l + j] = _mm512_sub_ps(_mm512_mul_ps(scale, _mm512_cvtepi32_ps(
                _mm512_and_si512(_mm512_srli_epi32(packed, 8 * j), low2))), minv);
    }
}

/* out[t * out_stride + r] = row r of a Q8_0 matrix times x[t * x_stride +
 * group(r) * group_in], for n tokens, optionally rounded to BF16. */
typedef struct {
    const uint8_t *data;
    const float *x;
    float *out;
    uint64_t x_stride, out_stride, in_dim, row_bytes, group_rows, group_in;
    uint32_t n;
    bool round;
} ds41_cpu_q8_batch_ctx;

static void ds41_cpu_q8_batch_worker(void *vctx, uint64_t first, uint64_t last) {
    const ds41_cpu_q8_batch_ctx *c = vctx;
    const __m512i offsets = ds41_cpu_row_offsets(c->row_bytes);
    __m512 w[32], acc[DS41_CPU_PASS];
    const float *xp[DS41_CPU_PASS];
    for (uint64_t tile = first; tile < last; tile++) {
        const uint64_t row = tile * 16;
        const uint8_t *rows = c->data + row * c->row_bytes;
        const float *x = c->x + row / c->group_rows * c->group_in;
        for (uint32_t t0 = 0; t0 < c->n; t0 += DS41_CPU_PASS) {
            const uint32_t nt = c->n - t0 < DS41_CPU_PASS ? c->n - t0 : DS41_CPU_PASS;
            for (uint32_t t = 0; t < nt; t++) {
                acc[t] = _mm512_setzero_ps();
                xp[t] = x + (uint64_t)(t0 + t) * c->x_stride;
            }
            for (uint64_t b = 0; b < c->in_dim / 32; b++) {
                ds41_cpu_q8_decode16(w, rows + b * 34, offsets);
                ds41_cpu_apply16(acc, w, xp, nt, b * 32, 32);
            }
            for (uint32_t t = 0; t < nt; t++)
                _mm512_storeu_ps(c->out + (uint64_t)(t0 + t) * c->out_stride + row,
                                 c->round ? ds41_cpu_bf16_16(acc[t]) : acc[t]);
        }
    }
}

/* F32 rows times many tokens with matvec_f32_worker's double chain for each
 * (row, token): (double)w * x added in input order, rounded to float at the
 * end.  SIMD lanes are eight tokens of xt, the inputs as [group][i][8]
 * doubles; four rows run at once for independent chains. */
typedef struct {
    const float *w, *x;
    double *xt;
    float *out;
    uint64_t in_dim, rows, x_stride, out_stride;
    uint32_t n, groups;
} ds41_cpu_f32_batch_ctx;

static void ds41_cpu_f32_transpose_worker(void *vctx, uint64_t first, uint64_t last) {
    const ds41_cpu_f32_batch_ctx *c = vctx;
    for (uint64_t g = first; g < last; g++) {
        double *xt = c->xt + g * c->in_dim * 8;
        for (uint32_t lane = 0; lane < 8; lane++) {
            const uint64_t t = g * 8 + lane;
            const float *x = c->x + t * c->x_stride;
            for (uint64_t i = 0; i < c->in_dim; i++)
                xt[i * 8 + lane] = t < c->n ? (double)x[i] : 0.0;
        }
    }
}

static void ds41_cpu_f32_batch_worker(void *vctx, uint64_t first, uint64_t last) {
    const ds41_cpu_f32_batch_ctx *c = vctx;
    for (uint64_t unit = first; unit < last; unit++) {
        const uint64_t g = unit % c->groups, r0 = unit / c->groups * 4;
        const uint64_t nr = c->rows - r0 < 4 ? c->rows - r0 : 4;
        const double *x = c->xt + g * c->in_dim * 8;
        const float *w0 = c->w + r0 * c->in_dim;
        const float *w1 = nr > 1 ? w0 + c->in_dim : w0;
        const float *w2 = nr > 2 ? w0 + 2 * c->in_dim : w0;
        const float *w3 = nr > 3 ? w0 + 3 * c->in_dim : w0;
        __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
        for (uint64_t i = 0; i < c->in_dim; i++) {
            const __m512d xv = _mm512_loadu_pd(x + i * 8);
            a0 = _mm512_add_pd(a0, _mm512_mul_pd(_mm512_set1_pd((double)w0[i]), xv));
            a1 = _mm512_add_pd(a1, _mm512_mul_pd(_mm512_set1_pd((double)w1[i]), xv));
            a2 = _mm512_add_pd(a2, _mm512_mul_pd(_mm512_set1_pd((double)w2[i]), xv));
            a3 = _mm512_add_pd(a3, _mm512_mul_pd(_mm512_set1_pd((double)w3[i]), xv));
        }
        double acc[4][8];
        _mm512_storeu_pd(acc[0], a0);
        _mm512_storeu_pd(acc[1], a1);
        _mm512_storeu_pd(acc[2], a2);
        _mm512_storeu_pd(acc[3], a3);
        for (uint64_t r = 0; r < nr; r++)
            for (uint64_t lane = 0; lane < 8 && g * 8 + lane < c->n; lane++)
                c->out[(g * 8 + lane) * c->out_stride + r0 + r] = (float)acc[r][lane];
    }
}

/* Routed experts of a token chunk grouped by expert: entries
 * [first[u], first[u + 1]) are token * 6 + slot pairs of the u-th distinct
 * expert, in token order.  mid and partial are indexed by entry. */
typedef struct {
    const uint8_t *gate[DS4_MAX_EXPERT], *up[DS4_MAX_EXPERT], *down[DS4_MAX_EXPERT];
    uint32_t first[DS4_MAX_EXPERT + 1];
    const uint32_t *entry;
    const float *x, *weight;
    float *mid, *partial;
    uint64_t x_stride, gate_row_bytes, up_row_bytes, down_row_bytes;
    float clamp;
} ds41_cpu_routed_batch_ctx;

static void ds41_cpu_routed_mid_batch_worker(void *vctx, uint64_t first, uint64_t last) {
    const ds41_cpu_routed_batch_ctx *c = vctx;
    const uint32_t tiles = DS4_N_FF_EXP / 16;
    const __m512i og = ds41_cpu_row_offsets(c->gate_row_bytes);
    const __m512i ou = ds41_cpu_row_offsets(c->up_row_bytes);
    __m512 wg[32], wu[32], ag[DS41_CPU_PASS], au[DS41_CPU_PASS];
    const float *xp[DS41_CPU_PASS];
    for (uint64_t unit = first; unit < last; unit++) {
        const uint32_t e = (uint32_t)(unit / tiles);
        const uint64_t row = (unit % tiles) * 16;
        const uint8_t *grows = c->gate[e] + row * c->gate_row_bytes;
        const uint8_t *urows = c->up[e] + row * c->up_row_bytes;
        for (uint32_t k0 = c->first[e]; k0 < c->first[e + 1]; k0 += DS41_CPU_PASS) {
            const uint32_t nt = c->first[e + 1] - k0 < DS41_CPU_PASS ?
                c->first[e + 1] - k0 : DS41_CPU_PASS;
            for (uint32_t t = 0; t < nt; t++) {
                ag[t] = au[t] = _mm512_setzero_ps();
                xp[t] = c->x + (uint64_t)(c->entry[k0 + t] / DS4_N_EXPERT_USED) * c->x_stride;
            }
            for (uint64_t b = 0; b < DS4_N_EMBD / QK_K; b++) {
                const uint8_t *gb = grows + b * sizeof(block_iq2_xxs);
                const uint8_t *ub = urows + b * sizeof(block_iq2_xxs);
                const __m512 dg = _mm512_mul_ps(_mm512_set1_ps(0.125f),
                    ds41_cpu_half16(_mm512_i32gather_epi32(og, gb, 1)));
                const __m512 du = _mm512_mul_ps(_mm512_set1_ps(0.125f),
                    ds41_cpu_half16(_mm512_i32gather_epi32(ou, ub, 1)));
                for (unsigned ib = 0; ib < QK_K / 32; ib++) {
                    ds41_cpu_iq2xxs_decode16(wg, gb, ib, dg, og);
                    ds41_cpu_iq2xxs_decode16(wu, ub, ib, du, ou);
                    ds41_cpu_apply16x2(ag, au, wg, wu, xp, nt, b * QK_K + ib * 32u, 32);
                }
            }
            for (uint32_t t = 0; t < nt; t++) {
                const uint32_t k = c->entry[k0 + t];
                float gate[16] __attribute__((aligned(64)));
                float up[16] __attribute__((aligned(64)));
                _mm512_storeu_ps(gate, ag[t]);
                _mm512_storeu_ps(up, au[t]);
                float *mid = c->mid + (uint64_t)k * DS4_N_FF_EXP + row;
                for (unsigned lane = 0; lane < 16; lane++) {
                    float g = gate[lane], u = up[lane];
                    if (c->clamp > 1.0e-6f) {
                        if (g > c->clamp) g = c->clamp;
                        if (u > c->clamp) u = c->clamp;
                        if (u < -c->clamp) u = -c->clamp;
                    }
                    mid[lane] = silu(g) * u * c->weight[k];
                }
            }
        }
    }
}

static void ds41_cpu_routed_down_batch_worker(void *vctx, uint64_t first, uint64_t last) {
    const ds41_cpu_routed_batch_ctx *c = vctx;
    const uint32_t tiles = DS4_N_EMBD / 16;
    const __m512i od = ds41_cpu_row_offsets(c->down_row_bytes);
    __m512 w[16], acc[DS41_CPU_PASS];
    const float *xp[DS41_CPU_PASS];
    for (uint64_t unit = first; unit < last; unit++) {
        const uint32_t e = (uint32_t)(unit / tiles);
        const uint64_t row = (unit % tiles) * 16;
        const uint8_t *rows = c->down[e] + row * c->down_row_bytes;
        for (uint32_t k0 = c->first[e]; k0 < c->first[e + 1]; k0 += DS41_CPU_PASS) {
            const uint32_t nt = c->first[e + 1] - k0 < DS41_CPU_PASS ?
                c->first[e + 1] - k0 : DS41_CPU_PASS;
            for (uint32_t t = 0; t < nt; t++) {
                acc[t] = _mm512_setzero_ps();
                xp[t] = c->mid + (uint64_t)c->entry[k0 + t] * DS4_N_FF_EXP;
            }
            for (uint64_t b = 0; b < DS4_N_FF_EXP / QK_K; b++) {
                const uint8_t *block = rows + b * sizeof(block_q2_K);
                const __m512i halves = _mm512_i32gather_epi32(
                    od, block + offsetof(block_q2_K, d), 1);
                const __m512 d = ds41_cpu_half16(halves);
                const __m512 dmin = ds41_cpu_half16(_mm512_srli_epi32(halves, 16));
                for (unsigned g = 0; g < QK_K / 16; g++) {
                    ds41_cpu_q2k_decode16(w, block, g, d, dmin, od);
                    ds41_cpu_apply16(acc, w, xp, nt, b * QK_K + g * 16u, 16);
                }
            }
            for (uint32_t t = 0; t < nt; t++)
                _mm512_storeu_ps(c->partial + (uint64_t)c->entry[k0 + t] * DS4_N_EMBD + row, acc[t]);
        }
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

/* n tokens through one Q8_0 matrix: out[t * out_stride + r] = row r times
 * x[t * x_stride + group(r) * group_in], rounded to BF16 when asked.  Every
 * value is bitwise ds41_cpu_matvec_q8_grouped's.  Returns false, having done
 * nothing, when the batched kernel does not apply. */
static bool ds41_cpu_matmul_q8(float *out, uint64_t out_stride, const ds4_model *m,
                               const ds4_tensor *t, const float *x, uint64_t x_stride,
                               uint32_t n, uint64_t group_rows, uint64_t group_in,
                               bool round) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (t->type != DS4_TENSOR_Q8_0 || t->ndim < 2 || t->dim[0] == 0 ||
        t->dim[0] % 32 || t->dim[0] > (uint64_t)(INT32_MAX / (15 * 34)) * 32) return false;
    const uint64_t rows = t->elements / t->dim[0];
    if (!rows || rows % 16 || !group_rows || group_rows % 16 || rows % group_rows) return false;
    ds41_cpu_q8_batch_ctx ctx = {
        .data = tensor_data(m, t), .x = x, .out = out, .x_stride = x_stride,
        .out_stride = out_stride, .in_dim = t->dim[0], .row_bytes = t->dim[0] / 32 * 34,
        .group_rows = group_rows, .group_in = group_in, .n = n, .round = round,
    };
    ds4_parallel_for_min_rows(rows / 16, ds41_cpu_q8_batch_worker, &ctx, 1);
    return true;
#else
    (void)out; (void)out_stride; (void)m; (void)t; (void)x; (void)x_stride;
    (void)n; (void)group_rows; (void)group_in; (void)round;
    return false;
#endif
}

/* n tokens through an F32 matrix, each output bitwise matvec_f32_worker's.
 * xt holds (n + 7) / 8 * 8 * in_dim doubles of scratch.  Returns false,
 * having done nothing, when the batched kernel does not apply. */
static bool ds41_cpu_matmul_f32(float *out, uint64_t out_stride, const ds4_model *m,
                                const ds4_tensor *t, const float *x, uint64_t x_stride,
                                uint32_t n, double *xt) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (t->type != DS4_TENSOR_F32 || t->ndim != 2 || !t->dim[0] || !t->dim[1] || !n)
        return false;
    ds41_cpu_f32_batch_ctx ctx = {
        .w = tensor_data(m, t), .x = x, .xt = xt, .out = out, .in_dim = t->dim[0],
        .rows = t->dim[1], .x_stride = x_stride, .out_stride = out_stride,
        .n = n, .groups = (n + 7) / 8,
    };
    ds4_parallel_for_min_rows(ctx.groups, ds41_cpu_f32_transpose_worker, &ctx, 1);
    ds4_parallel_for_min_rows((ctx.rows + 3) / 4 * ctx.groups, ds41_cpu_f32_batch_worker, &ctx, 1);
    return true;
#else
    (void)out; (void)out_stride; (void)m; (void)t; (void)x; (void)x_stride; (void)n; (void)xt;
    return false;
#endif
}

/* Routed experts of n tokens grouped by expert, for Q2 models.  partial gets
 * each (token, slot) down projection; the caller adds the slots in order.
 * Returns false, having done nothing, when the batched kernels do not apply. */
static bool ds41_cpu_routed_moe_batch(float *partial, float *mid, uint32_t *entry,
                                      const ds4_model *m, const ds4_layer_weights *l,
                                      const float *x, uint64_t x_stride, uint32_t n,
                                      const int *selected, const float *weights) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (!l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps ||
        l->ffn_gate_exps->type != DS4_TENSOR_IQ2_XXS ||
        l->ffn_up_exps->type != DS4_TENSOR_IQ2_XXS ||
        l->ffn_down_exps->type != DS4_TENSOR_Q2_K ||
        DS4_N_EXPERT > DS4_MAX_EXPERT || DS4_N_EXPERT_USED == 0 ||
        DS4_N_EXPERT_USED > DS4_MAX_EXPERT_USED ||
        DS4_N_EMBD % QK_K || DS4_N_FF_EXP % QK_K ||
        DS4_N_EMBD / QK_K > INT32_MAX / (15 * (int)sizeof(block_q2_K)) ||
        DS4_N_FF_EXP / QK_K > INT32_MAX / (15 * (int)sizeof(block_q2_K))) return false;
    ds41_cpu_routed_batch_ctx c = {0};
    uint32_t count[DS4_MAX_EXPERT] = {0}, used[DS4_MAX_EXPERT];
    const uint32_t pairs = n * DS4_N_EXPERT_USED;
    for (uint32_t k = 0; k < pairs; k++) {
        if (selected[k] < 0 || (uint32_t)selected[k] >= DS4_N_EXPERT)
            ds4_die("V4.1 CPU routed-MoE selected expert is outside range");
        count[selected[k]]++;
    }
    uint32_t n_used = 0, fill[DS4_MAX_EXPERT];
    c.first[0] = 0;
    for (uint32_t e = 0; e < DS4_N_EXPERT; e++) {
        if (!count[e]) continue;
        uint64_t gi, go, ui, uo, di, dout;
        c.gate[n_used] = tensor_expert_bytes(m, l->ffn_gate_exps, e, &gi, &go, &c.gate_row_bytes);
        c.up[n_used] = tensor_expert_bytes(m, l->ffn_up_exps, e, &ui, &uo, &c.up_row_bytes);
        c.down[n_used] = tensor_expert_bytes(m, l->ffn_down_exps, e, &di, &dout, &c.down_row_bytes);
        if (gi != DS4_N_EMBD || ui != DS4_N_EMBD || go != DS4_N_FF_EXP ||
            uo != DS4_N_FF_EXP || di != DS4_N_FF_EXP || dout != DS4_N_EMBD)
            ds4_die("V4.1 CPU routed-MoE found unexpected expert dimensions");
        used[e] = n_used;
        fill[n_used] = c.first[n_used];
        c.first[n_used + 1] = c.first[n_used] + count[e];
        n_used++;
    }
    for (uint32_t k = 0; k < pairs; k++) entry[fill[used[selected[k]]]++] = k;
    c.entry = entry;
    c.x = x;
    c.x_stride = x_stride;
    c.weight = weights;
    c.mid = mid;
    c.partial = partial;
    c.clamp = DS4_SWIGLU_CLAMP_EXP;
    /* Each expert is a NUMA group: a node takes the row half placed on it. */
    ds4_pool_dispatch((uint64_t)n_used * (DS4_N_FF_EXP / 16), ds41_cpu_routed_mid_batch_worker,
                      &c, DS4_N_FF_EXP / 16, 1);
    ds4_pool_dispatch((uint64_t)n_used * (DS4_N_EMBD / 16), ds41_cpu_routed_down_batch_worker,
                      &c, DS4_N_EMBD / 16, 1);
    return true;
#else
    (void)partial; (void)mid; (void)entry; (void)m; (void)l; (void)x; (void)x_stride;
    (void)n; (void)selected; (void)weights;
    return false;
#endif
}

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* The layouts the routed row kernels cover: Q4_K experts, or the Q2 models'
 * IQ2_XXS gate/up and Q2_K down, both one row per SIMD lane. */
static bool ds41_cpu_routed_rows(const ds4_layer_weights *l) {
    const bool q4 = l->ffn_gate_exps && l->ffn_up_exps && l->ffn_down_exps &&
        l->ffn_gate_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_up_exps->type == DS4_TENSOR_Q4_K &&
        l->ffn_down_exps->type == DS4_TENSOR_Q4_K;
    const bool q2 = l->ffn_gate_exps && l->ffn_up_exps && l->ffn_down_exps &&
        l->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
        l->ffn_up_exps->type == DS4_TENSOR_IQ2_XXS &&
        l->ffn_down_exps->type == DS4_TENSOR_Q2_K;
    return (q4 || q2) &&
        DS4_N_EXPERT_USED > 0 && DS4_N_EXPERT_USED <= DS4_MAX_EXPERT_USED &&
        DS4_N_EMBD > 0 && DS4_N_FF_EXP > 0 &&
        DS4_N_EMBD % QK_K == 0 && DS4_N_FF_EXP % QK_K == 0 &&
        DS4_N_EMBD / QK_K <= INT32_MAX / (15 * (int)sizeof(block_q4_K)) &&
        DS4_N_FF_EXP / QK_K <= INT32_MAX / (15 * (int)sizeof(block_q4_K));
}

/* Gate/up into mid (expert weights applied), then the down rows into out. */
static void ds41_cpu_routed_init(glm_routed_moe_f32_mid_ctx *mc,
                                glm_routed_moe_f32_down_ctx *dc,
                                const ds4_layer_weights *l, const float *x,
                                float *mid, float *out) {
    *mc = (glm_routed_moe_f32_mid_ctx){
        .mid = mid, .x = x, .in_dim = DS4_N_EMBD, .out_dim = DS4_N_FF_EXP,
        .gate_type = l->ffn_gate_exps->type, .up_type = l->ffn_up_exps->type,
        .n_expert = DS4_N_EXPERT_USED, .clamp = DS4_SWIGLU_CLAMP_EXP,
    };
    *dc = (glm_routed_moe_f32_down_ctx){
        .out = out, .mid = mid, .in_dim = DS4_N_FF_EXP, .out_dim = DS4_N_EMBD,
        .down_type = l->ffn_down_exps->type, .n_expert = DS4_N_EXPERT_USED,
    };
    /* The IQ2_XXS pair kernel's code table, before any dispatch runs the
     * mid rows; the dispatch publishes it to the workers. */
    if (l->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS)
        pthread_once(&ds41_iq2xxs_once, ds41_iq2xxs_tables_init);
}

/* Point the contexts at the rows of the selected experts. */
static void ds41_cpu_routed_bind(glm_routed_moe_f32_mid_ctx *mc,
                                glm_routed_moe_f32_down_ctx *dc,
                                const ds4_model *m, const ds4_layer_weights *l,
                                const int *selected, const float *weights) {
    for (uint32_t slot = 0; slot < DS4_N_EXPERT_USED; slot++) {
        uint64_t gi, go, ui, uo, di, dout;
        if (selected[slot] < 0 || (uint32_t)selected[slot] >= DS4_N_EXPERT)
            ds4_die("V4.1 CPU routed-MoE selected expert is outside range");
        mc->gate_base[slot] = tensor_expert_bytes(m, l->ffn_gate_exps,
            (uint32_t)selected[slot], &gi, &go, &mc->gate_row_bytes[slot]);
        mc->up_base[slot] = tensor_expert_bytes(m, l->ffn_up_exps,
            (uint32_t)selected[slot], &ui, &uo, &mc->up_row_bytes[slot]);
        dc->down_base[slot] = tensor_expert_bytes(m, l->ffn_down_exps,
            (uint32_t)selected[slot], &di, &dout, &dc->down_row_bytes[slot]);
        if (gi != mc->in_dim || ui != mc->in_dim || go != mc->out_dim ||
            uo != mc->out_dim || di != dc->in_dim || dout != dc->out_dim)
            ds4_die("V4.1 CPU routed-MoE found unexpected expert dimensions");
        mc->expert_weight[slot] = weights[slot];
    }
}
#endif

static void ds41_cpu_routed_moe(float *out, float *mid, const ds4_model *m,
                               const ds4_layer_weights *l, const float *x,
                               const int *selected, const float *weights) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    if (ds41_cpu_routed_rows(l)) {
        glm_routed_moe_f32_mid_ctx mc;
        glm_routed_moe_f32_down_ctx dc;
        ds41_cpu_routed_init(&mc, &dc, l, x, mid, out);
        ds41_cpu_routed_bind(&mc, &dc, m, l, selected, weights);
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
