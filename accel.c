// accel.c — AVX2+FMA SIMD accelerator for quantized matmul
// Part of deepseek.go — github.com/ariannamethod/deepseek.go
// Linked via CGO. Compiled with -mavx2 -mfma -O3.

#pragma GCC target("avx2,fma")
#include "accel.h"
#include <immintrin.h>
#include <string.h>

// ============================================================
// FP16 → FP32 conversion
// ============================================================

static inline float fp16_to_fp32(const uint8_t* p) {
    uint16_t h;
    memcpy(&h, p, 2);
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        // Denormalized: normalize by shifting mantissa up
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
        exp = exp - 15 + 127;
        uint32_t bits = sign | (exp << 23) | (mant << 13);
        float f; memcpy(&f, &bits, 4); return f;
    }
    if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float f; memcpy(&f, &bits, 4); return f;
    }
    exp = exp - 15 + 127;
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

// ============================================================
// Horizontal sum of __m256
// ============================================================

static inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s4 = _mm_add_ps(hi, lo);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}

// ============================================================
// Q4_K: 256 elements per block, 144 bytes
// d(fp16) dmin(fp16) scales[12] qs[128]
// ============================================================

#define Q4K_BLK 256
#define Q4K_BPB 144

static inline void get_scale_min_k4(int j, const uint8_t* sc, uint8_t* scale, uint8_t* mn) {
    if (j < 4) {
        *scale = sc[j] & 63;
        *mn = sc[j+4] & 63;
    } else {
        *scale = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
        *mn = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

static float dot_q4k(const uint8_t* w, const float* x, int cols) {
    int nblk = cols / Q4K_BLK;
    float total = 0.0f;

    for (int b = 0; b < nblk; b++) {
        const uint8_t* blk = w + b * Q4K_BPB;
        float d = fp16_to_fp32(blk);
        float dmin = fp16_to_fp32(blk + 2);
        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;
        const float* xp = x + b * Q4K_BLK;

        int qi = 0, is = 0;

        for (int j = 0; j < Q4K_BLK; j += 64) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(is, scales, &sc0, &m0);
            get_scale_min_k4(is+1, scales, &sc1, &m1);

            // Low nibble: 32 elements → xp[j..j+31]
            __m256 vdot = _mm256_setzero_ps();
            __m256 vsx  = _mm256_setzero_ps();

            for (int l = 0; l < 32; l += 8) {
                __m128i raw = _mm_loadl_epi64((const __m128i*)(qs + qi + l));
                __m256i wide = _mm256_cvtepu8_epi32(raw);
                __m256i lo = _mm256_and_si256(wide, _mm256_set1_epi32(0x0F));
                __m256 fq = _mm256_cvtepi32_ps(lo);
                __m256 vx = _mm256_loadu_ps(xp + j + l);
                vdot = _mm256_fmadd_ps(fq, vx, vdot);
                vsx  = _mm256_add_ps(vx, vsx);
            }

            total += d * (float)sc0 * hsum256(vdot) - dmin * (float)m0 * hsum256(vsx);

            // High nibble: 32 elements → xp[j+32..j+63]
            vdot = _mm256_setzero_ps();
            vsx  = _mm256_setzero_ps();

            for (int l = 0; l < 32; l += 8) {
                __m128i raw = _mm_loadl_epi64((const __m128i*)(qs + qi + l));
                __m256i wide = _mm256_cvtepu8_epi32(raw);
                __m256i hi = _mm256_srli_epi32(wide, 4);
                __m256 fq = _mm256_cvtepi32_ps(hi);
                __m256 vx = _mm256_loadu_ps(xp + j + 32 + l);
                vdot = _mm256_fmadd_ps(fq, vx, vdot);
                vsx  = _mm256_add_ps(vx, vsx);
            }

            total += d * (float)sc1 * hsum256(vdot) - dmin * (float)m1 * hsum256(vsx);

            qi += 32;
            is += 2;
        }
    }
    return total;
}

void accel_matmul_q4k_range(float* out, const uint8_t* w, const float* x,
                             int start, int end, int cols) {
    int bpr = (cols / Q4K_BLK) * Q4K_BPB;
    for (int r = start; r < end; r++) {
        out[r] = dot_q4k(w + r * bpr, x, cols);
    }
}

// ============================================================
// Q6_K: 256 elements per block, 210 bytes
// ql[128] qh[64] scales[16] d(fp16)
// ============================================================

#define Q6K_BLK 256
#define Q6K_BPB 210

static float dot_q6k(const uint8_t* w, const float* x, int cols) {
    int nblk = cols / Q6K_BLK;
    float sum = 0.0f;

    for (int b = 0; b < nblk; b++) {
        const uint8_t* blk = w + b * Q6K_BPB;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* scales = (const int8_t*)(blk + 192);
        float d = fp16_to_fp32(blk + 208);

        const float* xp = x + b * Q6K_BLK;

        for (int n128 = 0; n128 < 2; n128++) {
            const uint8_t* qlP = ql + n128 * 64;
            const uint8_t* qhP = qh + n128 * 32;
            const int8_t* scP = scales + n128 * 8;
            const float* xB = xp + n128 * 128;

            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (qlP[l] & 0x0F)    | (((qhP[l] >> 0) & 3) << 4);
                int q2 = (qlP[l+32] & 0x0F) | (((qhP[l] >> 2) & 3) << 4);
                int q3 = (qlP[l] >> 4)       | (((qhP[l] >> 4) & 3) << 4);
                int q4 = (qlP[l+32] >> 4)    | (((qhP[l] >> 6) & 3) << 4);

                float s0 = d * (float)scP[is + 0];
                float s2 = d * (float)scP[is + 2];
                float s4 = d * (float)scP[is + 4];
                float s6 = d * (float)scP[is + 6];

                sum += s0 * (float)(q1 - 32) * xB[l];
                sum += s2 * (float)(q2 - 32) * xB[l + 32];
                sum += s4 * (float)(q3 - 32) * xB[l + 64];
                sum += s6 * (float)(q4 - 32) * xB[l + 96];
            }
        }
    }
    return sum;
}

void accel_matmul_q6k_range(float* out, const uint8_t* w, const float* x,
                             int start, int end, int cols) {
    int bpr = (cols / Q6K_BLK) * Q6K_BPB;
    for (int r = start; r < end; r++) {
        out[r] = dot_q6k(w + r * bpr, x, cols);
    }
}

// ============================================================
// Q4_0: 32 elements per block, 18 bytes
// d(fp16) qs[16]
// ============================================================

#define Q40_BLK 32
#define Q40_BPB 18

static float dot_q4_0(const uint8_t* w, const float* x, int cols) {
    int nblk = cols / Q40_BLK;
    float total = 0.0f;

    for (int b = 0; b < nblk; b++) {
        const uint8_t* blk = w + b * Q40_BPB;
        float d = fp16_to_fp32(blk);
        const uint8_t* qs = blk + 2;
        const float* xp = x + b * Q40_BLK;

        __m256 vdot = _mm256_setzero_ps();
        __m256i v8 = _mm256_set1_epi32(8);

        // Low nibble: positions 0..15
        for (int l = 0; l < 16; l += 8) {
            __m128i raw = _mm_loadl_epi64((const __m128i*)(qs + l));
            __m256i wide = _mm256_cvtepu8_epi32(raw);
            __m256i lo = _mm256_sub_epi32(_mm256_and_si256(wide, _mm256_set1_epi32(0x0F)), v8);
            __m256 fq = _mm256_cvtepi32_ps(lo);
            __m256 vx = _mm256_loadu_ps(xp + l);
            vdot = _mm256_fmadd_ps(fq, vx, vdot);
        }

        // High nibble: positions 16..31
        for (int l = 0; l < 16; l += 8) {
            __m128i raw = _mm_loadl_epi64((const __m128i*)(qs + l));
            __m256i wide = _mm256_cvtepu8_epi32(raw);
            __m256i hi = _mm256_sub_epi32(_mm256_srli_epi32(wide, 4), v8);
            __m256 fq = _mm256_cvtepi32_ps(hi);
            __m256 vx = _mm256_loadu_ps(xp + 16 + l);
            vdot = _mm256_fmadd_ps(fq, vx, vdot);
        }

        total += d * hsum256(vdot);
    }
    return total;
}

void accel_matmul_q4_0_range(float* out, const uint8_t* w, const float* x,
                              int start, int end, int cols) {
    int bpr = (cols / Q40_BLK) * Q40_BPB;
    for (int r = start; r < end; r++) {
        out[r] = dot_q4_0(w + r * bpr, x, cols);
    }
}

// ============================================================
// Q8_0: 32 elements per block, 34 bytes
// d(fp16) qs[32] (int8)
// ============================================================

#define Q80_BLK 32
#define Q80_BPB 34

static float dot_q8_0(const uint8_t* w, const float* x, int cols) {
    int nblk = cols / Q80_BLK;
    float total = 0.0f;

    for (int b = 0; b < nblk; b++) {
        const uint8_t* blk = w + b * Q80_BPB;
        float d = fp16_to_fp32(blk);
        const int8_t* qs = (const int8_t*)(blk + 2);
        const float* xp = x + b * Q80_BLK;

        __m256 vdot = _mm256_setzero_ps();

        for (int l = 0; l < 32; l += 8) {
            __m128i raw = _mm_loadl_epi64((const __m128i*)(qs + l));
            __m256i wide = _mm256_cvtepi8_epi32(raw);
            __m256 fq = _mm256_cvtepi32_ps(wide);
            __m256 vx = _mm256_loadu_ps(xp + l);
            vdot = _mm256_fmadd_ps(fq, vx, vdot);
        }

        total += d * hsum256(vdot);
    }
    return total;
}

void accel_matmul_q8_0_range(float* out, const uint8_t* w, const float* x,
                              int start, int end, int cols) {
    int bpr = (cols / Q80_BLK) * Q80_BPB;
    for (int r = start; r < end; r++) {
        out[r] = dot_q8_0(w + r * bpr, x, cols);
    }
}
