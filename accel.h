// accel.h — AVX2+FMA SIMD accelerator for quantized matmul
// Part of deepseek.go — github.com/ariannamethod/deepseek.go
#ifndef ACCEL_H
#define ACCEL_H

#include <stdint.h>

// Q4_K matmul: out[start..end) = W_q4k[start..end, cols] @ x[cols]
void accel_matmul_q4k_range(float* out, const uint8_t* w, const float* x,
                             int start, int end, int cols);

// Q6_K matmul: out[start..end) = W_q6k[start..end, cols] @ x[cols]
void accel_matmul_q6k_range(float* out, const uint8_t* w, const float* x,
                             int start, int end, int cols);

// Q4_0 matmul: out[start..end) = W_q4_0[start..end, cols] @ x[cols]
void accel_matmul_q4_0_range(float* out, const uint8_t* w, const float* x,
                              int start, int end, int cols);

// Q8_0 matmul: out[start..end) = W_q8_0[start..end, cols] @ x[cols]
void accel_matmul_q8_0_range(float* out, const uint8_t* w, const float* x,
                              int start, int end, int cols);

#endif // ACCEL_H
