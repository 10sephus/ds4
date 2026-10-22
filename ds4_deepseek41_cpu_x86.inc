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
        if (q2) pthread_once(&ds41_iq2xxs_once, ds41_iq2xxs_tables_init);
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
