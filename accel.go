package main

// accel.go — CGO wrapper for AVX2+FMA SIMD accelerator
// Calls into accel.c for quantized matmul. Go handles threading via goroutines.

/*
#cgo CFLAGS: -O3
#include "accel.h"
*/
import "C"
import (
	"sync"
	"unsafe"
)

// AccelMatMulQ4K dispatches Q4_K matmul to AVX2+FMA C code
func AccelMatMulQ4K(out []float32, w []byte, x []float32, rows, cols int) {
	if rows < numWorkers*4 {
		C.accel_matmul_q4k_range(
			(*C.float)(unsafe.Pointer(&out[0])),
			(*C.uint8_t)(unsafe.Pointer(&w[0])),
			(*C.float)(unsafe.Pointer(&x[0])),
			C.int(0), C.int(rows), C.int(cols))
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			C.accel_matmul_q4k_range(
				(*C.float)(unsafe.Pointer(&out[0])),
				(*C.uint8_t)(unsafe.Pointer(&w[0])),
				(*C.float)(unsafe.Pointer(&x[0])),
				C.int(s), C.int(e), C.int(cols))
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

// AccelMatMulQ6K dispatches Q6_K matmul to AVX2+FMA C code
func AccelMatMulQ6K(out []float32, w []byte, x []float32, rows, cols int) {
	if rows < numWorkers*4 {
		C.accel_matmul_q6k_range(
			(*C.float)(unsafe.Pointer(&out[0])),
			(*C.uint8_t)(unsafe.Pointer(&w[0])),
			(*C.float)(unsafe.Pointer(&x[0])),
			C.int(0), C.int(rows), C.int(cols))
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			C.accel_matmul_q6k_range(
				(*C.float)(unsafe.Pointer(&out[0])),
				(*C.uint8_t)(unsafe.Pointer(&w[0])),
				(*C.float)(unsafe.Pointer(&x[0])),
				C.int(s), C.int(e), C.int(cols))
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

// AccelMatMulQ4_0 dispatches Q4_0 matmul to AVX2+FMA C code
func AccelMatMulQ4_0(out []float32, w []byte, x []float32, rows, cols int) {
	if rows < numWorkers*4 {
		C.accel_matmul_q4_0_range(
			(*C.float)(unsafe.Pointer(&out[0])),
			(*C.uint8_t)(unsafe.Pointer(&w[0])),
			(*C.float)(unsafe.Pointer(&x[0])),
			C.int(0), C.int(rows), C.int(cols))
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			C.accel_matmul_q4_0_range(
				(*C.float)(unsafe.Pointer(&out[0])),
				(*C.uint8_t)(unsafe.Pointer(&w[0])),
				(*C.float)(unsafe.Pointer(&x[0])),
				C.int(s), C.int(e), C.int(cols))
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

// AccelMatMulQ8_0 dispatches Q8_0 matmul to AVX2+FMA C code
func AccelMatMulQ8_0(out []float32, w []byte, x []float32, rows, cols int) {
	if rows < numWorkers*4 {
		C.accel_matmul_q8_0_range(
			(*C.float)(unsafe.Pointer(&out[0])),
			(*C.uint8_t)(unsafe.Pointer(&w[0])),
			(*C.float)(unsafe.Pointer(&x[0])),
			C.int(0), C.int(rows), C.int(cols))
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers

	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			C.accel_matmul_q8_0_range(
				(*C.float)(unsafe.Pointer(&out[0])),
				(*C.uint8_t)(unsafe.Pointer(&w[0])),
				(*C.float)(unsafe.Pointer(&x[0])),
				C.int(s), C.int(e), C.int(cols))
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

